from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from typing import List, Dict, Any

text_splitter = RecursiveCharacterTextSplitter( 
    chunk_size=1000,
    chunk_overlap=250,
    separators=[
        "\n\n", ". ", "! ", "? ", "\n", ",", ";", " ", "```",
        "\n\n## ", "\n\n### ", "\n\n#### ", "\n\n• ",
        "## ", "### ", "#### ", "• "
    ], 
    length_function=len
)

def process_text(text: str, page_num: int, source_path: str) -> List[Dict[str, Any]]:
    """
    Splits text into chunks with associated metadata
    
    Args:
        text: Raw text content to process
        page_num: Zero-based page number
        source_path: Original file path of the source document
        
    Returns:
        List of text chunks with metadata
    """
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

def process_generic_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Processes non-PDF/audio/image files into text chunks
    
    Args:
        filepath: Path to the file to process
        
    Returns:
        List of text chunks with metadata
    """
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