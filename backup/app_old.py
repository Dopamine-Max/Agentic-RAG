import os
import shutil
import pymupdf
#manually installed through pip due to issue building wheel in poetry 
import whisper # poetry run pip install git+https://github.com/openai/whisper.git@v20231117
import mimetypes
import PIL.Image
import hashlib
import asyncio
from typing import List
from google.genai import Client
from qdrant_client import QdrantClient, models
from googlesearch import search
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from config import GEMINI_API_KEY, QDRANT_API_KEY, QDRANT_URL
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

OLLAMA_BASE_URL = "https://981c-104-154-150-250.ngrok-free.app" #does not work in .env file

gemini_client = Client(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(
    url = QDRANT_URL,
    api_key=QDRANT_API_KEY
)
transcribe_model = whisper.load_model("base")
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)
text_splitter = RecursiveCharacterTextSplitter( 
    chunk_size=1000,
    chunk_overlap=250,
    separators=["\n\n", ". ", "! ", "? ", "\n",",",";", " ","```","\n\n## ", "\n\n### ", "\n\n#### ", "\n\n• ", "## ", "### ", "#### ", "• "], 
    length_function=len
)

def initialize_vector_store():
    try:
        qdrant_client.recreate_collection(
            collection_name="rag_embeddings",
            vectors_config=models.VectorParams(
                size=768,  # Match nomic-embed-text dimension
                distance=models.Distance.COSINE
            )
        )
    except Exception as e:
        print(f"Collection already exists: {str(e)}")

def handle_file(filepath):
    mime_type, _ = mimetypes.guess_type(filepath)
    items = []
    
    if not mime_type:
        items = process_generic_file(filepath)
    elif mime_type.startswith('application/pdf'):
        items = process_pdf(filepath)
    elif mime_type.startswith('audio'):
        items = process_audio(filepath)
    elif mime_type.startswith('image'):
        items = process_image(filepath)
    else:
        items = process_generic_file(filepath)
    
    return items


# Save tables
def process_pdf_tables(page_num, filepath):
    try:
        import tabula #issue with preemptive tabula processing
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
        if not tables:
            return []
        return [{
            "type": "table",
            "text": "\n".join([" | ".join(map(str, row)) for row in table.values]),
            "metadata": {
                "source": filepath,
                "page": page_num + 1,
                "segment_type": "table"
            }
        } for table in tables]
    except Exception as e:
        print(f"Table processing error: {str(e)}")
        return []
        
# Save text chunks
def process_text(text, page_num, source_path):
    chunks = text_splitter.split_text(text)
    return [{
        "type": "text",
        "text": chunk,
        "metadata": {
            "source": source_path,
            "page": page_num + 1,
            "segment_type": "text"
        }
    } for chunk in chunks]
        
# Save images
def process_pdf_images(doc, page, page_num):
    
    os.makedirs("images", exist_ok=True)
    items = []
    
    for img_idx, img in enumerate(page.get_images()):
        try:
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            
            # Save PDF-extracted image
            image_name = f"images/pdf_{os.path.basename(doc.name)}_p{page_num}_i{img_idx}.png"
            pix.save(image_name)
            
            # Get description from saved image file
            description = describe_image(image_name)
            
            items.append({
                "type": "image",
                "text": description,
                "metadata": {
                    "source": doc.name,
                    "page": page_num + 1,
                    "segment_type": "image",
                    "image_path": image_name
                }
            })
        except Exception as e:
            print(f"PDF image processing error: {str(e)}")
    
    return items
        
def process_pdf(filepath):
    
    doc = pymupdf.open(filepath)
    num_pages=len(doc)
    items = []
    
    for page_num in range(num_pages):
        
        page = doc[page_num]
        page_items = []
        
        # Process text
        text = page.get_text()
        page_items.extend(process_text(text, page_num, filepath))
        
        # Process tables
        page_items.extend(process_pdf_tables(page_num, filepath))
        
        # Process images
        page_items.extend(process_pdf_images(doc, page, page_num))
        
        items.extend(page_items)
    
    return items
        
def process_audio(filepath):
    try:
        result = transcribe_model.transcribe(filepath)
        return [{
            "type": "audio",
            "text": chunk,
            "metadata": {
                "source": filepath,
                "segment_type": "audio"
            }
        } for chunk in text_splitter.split_text(result["text"])]
    except Exception as e:
        print(f"Audio processing failed: {str(e)}")
        return []

def process_image(filepath):
    try:
        os.makedirs("images", exist_ok=True)
        # Copy user-uploaded image to images directory
        base_name = os.path.basename(filepath)
        image_name = f"images/uploaded_{base_name}"
        
        with open(filepath, 'rb') as src, open(image_name, 'wb') as dst:
            dst.write(src.read())
        
        # Get description from saved copy
        return [{
            "type": "image",
            "text": describe_image(image_name),
            "metadata": {
                "source": image_name,
                "segment_type": "image"
            }
        }]
    except Exception as e:
        print(f"Image processing error: {str(e)}")
        return []

    
def process_generic_file(filepath):
    try:
        docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
        return [{
            "type": "text",
            "text": chunk,
            "metadata": {
                "source": filepath,
                "segment_type": "text"
            }
        } for doc in docs for chunk in text_splitter.split_text(doc.text)]
    except Exception as e:
        print(f"Generic file processing failed: {str(e)}")
        return []
    
def describe_image(image_path):
    try:
        # Define a custom system prompt for image description
        system_prompt = (
            "You are an expert at analyzing and describing images. "
            "Provide a detailed description of the image, including key objects, colors, layout and any and all text"
        )
        
        with PIL.Image.open(image_path) as img:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[system_prompt, img],
            )
            
            return response.text
        
    except Exception as e:
        print(f"Image description failed: {str(e)}")
        return "No description available"
    
def generate_file_embeddings(file_paths: list):
    
    initialize_vector_store()
    
    for file_path in file_paths:
        items = handle_file(file_path)
        for item in items:
            store_embedding(
                text=item["text"],
                metadata=item["metadata"],
                content_type=item["type"]
            )
            
def store_embedding(text, metadata, content_type):
    
    try:
        # Add default score based on content type
        base_scores = {
            "text": 1.0,
            "table": 0.95,
            "image": 0.9,
            "audio": 0.85
        }
        
        metadata["base_score"] = base_scores.get(content_type)
        
        
        # Generate embedding
        embedding = embed_model.get_text_embedding(text)
        
        # Create unique ID from text hash (fixed encoding)
        hash_input = (text + str(metadata)).encode('utf-8')  # Encode to bytes
        point_id = hashlib.md5(hash_input).hexdigest()  # Now using bytes input
        
        # Upsert to Qdrant
        qdrant_client.upsert(
            collection_name="rag_embeddings",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "type": content_type,
                        "metadata": metadata,
                        "base_score": metadata["base_score"]
                    }
                )
            ]
        )
        
        record = qdrant_client.retrieve(
        collection_name="rag_embeddings",
        ids=[point_id]
        )
        assert record, "Embedding storage failed"
        
    except Exception as e:
        print(f"Failed to store embedding: {str(e)}")
        
def cleanup_resources(perform_cleanup: bool = False):
    
    if not perform_cleanup:
        return
    
    image_dir = "images"
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
                
    collection_name = "rag_embeddings"
            
    try:

        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=models.PointsSelector(
                points=models.AllSelector()
            )
        )
        print(f"Cleared all data from '{collection_name}'")
        
    except Exception as e:
        
        print(f"Qdrant cleanup failed: {str(e)}")
        
def get_urls(query):
    try:
        results = search(
            query,
            num_results=10,
            advanced=True,
        )
        result_processed = [
            result.url for result in results 
            if result.url.startswith(('http://', 'https://'))
            and not result.url.endswith(('.pdf', '.doc', '.docx'))]
        return result_processed[:5]
    
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return []
    
async def crawl_parallel(urls, query,  max_concurrent=5):
    
    bm25_filter = BM25ContentFilter(
        user_query=query,
        # Adjust for stricter or looser results
        bm25_threshold=1.25  
    )
    
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
    
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        user_agent_mode="random"
    )
    
    crawl_config = CrawlerRunConfig(
        excluded_tags=[
            'nav', 'header', 'footer', 'script', 'style', 
            'form', 'aside', 'sidebar', 'menu', 'button'
        ],
        markdown_generator=md_generator,
        cache_mode=CacheMode.BYPASS,
        exclude_external_images=True
    )

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                try:
                    async with asyncio.timeout(10):
                        result = await crawler.arun(
                            url=url,
                            config=crawl_config,
                            extracted_metadata=["title", "description"],
                            session_id="session1"
                        )
                    
                    if result.success:
                        print(f"Crawled: {url}")
                        return {
                            "url": url,
                            "content": result.markdown.fit_markdown,
                            "title": result.metadata.get("title"),
                            "description": result.metadata.get("description")
                        }
                    else:
                        print(f"Failed: {url} - {result.error_message}")
                        return None
                
                except asyncio.TimeoutError:
                    print(f"Timeout crawling: {url}")
                    return None
                
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    return None

        results = await asyncio.gather(*[process_url(url) for url in urls])
        return results
    
    finally:
        await crawler.close()
        
    
def process_web_search(query):
    urls = get_urls(query)  # Fast URL fetch
    web_data = asyncio.run(crawl_parallel(urls, query=query))
    processed = []
    for item in web_data:
        if item is None or not item.get('content'):
            continue
        
        
        for chunk in text_splitter.split_text(item['content']):
            processed.append({
                "text": chunk,
                "score": 0.85,  # Slightly below DB results
                "metadata": {
                    "source": "web",
                    "url": item['url']
                }
            })
        
    return processed
        

def prioritize_content(items, max_tokens):
    selected = []
    current_tokens = 0
    
    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        item_tokens = len(item["text"]) // 4  # Approx token count
        if current_tokens + item_tokens > max_tokens:
            break
        selected.append(item)
        current_tokens += item_tokens
    
    return selected

def retrieve_from_qdrant(query, top_k):
    try:
        return qdrant_client.search(
            collection_name="rag_embeddings",
            query_vector=embed_model.get_text_embedding(query),
            limit=top_k
        )
    except Exception as e:
        print(f"Error retrieving from Qdrant: {str(e)}")
        return []
    
def retrieve_context(query, use_web):
    # Always start with vector DB results
    db_results = retrieve_from_qdrant(query, top_k=5 if use_web else 10)
    db_content = [
        {
            "text": item.payload["text"],
            "score": item.score,
            "metadata": item.payload["metadata"]
        }
        for item in db_results
    ]
    
    if not use_web:
        return db_content
    
    # Get fresh web content but DON'T store it
    web_content = process_web_search(query)
    
    # Combine using smart truncation
    return prioritize_content(
        db_content + web_content,
        max_tokens=3500  # Leave room for prompt
    )
    
def generate_response(query, context, model_name):
    try:
        # Format context with sources
        if hasattr(context[0], 'payload'):  # Qdrant SearchResults
            context_str = "\n\n".join(
                [f"• [Source: {chunk.payload['metadata'].get('source', 'unknown')}]: "
                 f"{chunk.payload['text']}" 
                 for i, chunk in enumerate(context)]
            )
        else:  # Dictionary format
            context_str = "\n\n".join(
                [f"• [Source: {chunk['metadata'].get('source', 'unknown')}]: "
                 f"{chunk['text']}" 
                 for i, chunk in enumerate(context)]
            )
        
        if not context:
            return "No relevant information found. Please try another query."

        # Create the Ollama client
        llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7,
            system="You are an analytical assistant that synthesizes information from multiple sources. " \
                "Always cite sources using metadata from context. Example: " \
                "'According to [Source Name]/(source URL)...'. " \
                "If unsure, say 'Based on my analysis of available information...'"
        )

        # Construct the prompt
        messages = [
            SystemMessage(content="Analyze the following context and provide a reasoned response:"),
            HumanMessage(content=f"Query: {query}\n\n\nContext:\n{context_str}")
        ]
        

        # Generate response
        response = llm.invoke(messages)
        
        return response.content

    except Exception as e:
        print(f"Generation failed: {str(e)}")
        return "I encountered an error while processing your request."
    
def full_qa_pipeline(query, use_web = True):
    # Retrieve context
    context = retrieve_context(query, use_web)
    print(context)

    # Generate response
    return generate_response(query, context, "deepseek-r1:8b")
