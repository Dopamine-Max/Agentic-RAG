import os
import shutil
import hashlib
from typing import Dict, List, Any
from qdrant_client import QdrantClient, models
from llama_index.embeddings.ollama import OllamaEmbedding
from config import OLLAMA_BASE_URL
from config import QDRANT_URL, QDRANT_API_KEY

# Initialize embedding model here
embed_model = OllamaEmbedding(  
    model_name="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

# Re-Initialize Qdrant here (required due to issue passing between modules)
qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

def store_embedding(text: str, metadata: Dict[str, Any], content_type: str) -> None:
    """
    Stores text embedding in Qdrant with associated metadata.
    
    Args:
        text: The text content to embed and store
        metadata: Document metadata including source information
        content_type: Type of content (text/image/audio)
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    try:
    
        
        embedding = embed_model.get_text_embedding(text)
        
        # Create unique ID from content hash
        hash_input = (text + str(metadata)).encode('utf-8')
        point_id = hashlib.md5(hash_input).hexdigest()
        
        qdrant_client.upsert(
            collection_name="rag_embeddings",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "type": content_type,
                        "metadata": metadata
                    }
                )
            ]
        )
        
    except Exception as e:
        print(f"Failed to store embedding: {str(e)}")
        raise

def retrieve_from_qdrant(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves similar content from Qdrant vector store.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        
    Returns:
        List of search results with scores and payloads
    """
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    
    try:
        query_embedding = embed_model.get_text_embedding(query)
        
        # Perform search
        results = qdrant_client.search(
            collection_name="rag_embeddings",
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [hit.payload for hit in results]
    
    except Exception as e:
        print(f"Retrieval failed: {str(e)}")
        return []
    
def cleanup_resources(perform_cleanup: bool = False) -> None:
    """
    Cleans up both vector store data and local image resources
    
    Args:
        perform_cleanup: Flag to actually perform cleanup when True
    """
    if not perform_cleanup:
        return

    # Clean vector store
    try:
        if qdrant_client.collection_exists("rag_embeddings"):
            qdrant_client.delete_collection(collection_name="rag_embeddings")
            print("Vector store data cleared successfully")
    except Exception as e:
        print(f"Vector store cleanup failed: {str(e)}")
        raise

    # Clean image directory
    image_dir = "images"
    try:
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
            print("Image resources cleaned successfully")
    except Exception as e:
        print(f"Image cleanup failed: {str(e)}")
        raise