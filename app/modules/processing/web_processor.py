import requests
from typing import List, Dict, Any, Optional
from config import FAST_API_ENDPOINT
from modules.vector_store.operations import store_embedding
from modules.processing.generic_processor import text_splitter

def process_web_search(query: str) -> List[Dict[str, Any]]:
    """
    Executes parallel web search and content processing.
    
    Args:
        query: Search query to execute
        
    Returns:
        List of processed web content chunks
    """
    try:
        web_data = requests.post(
            FAST_API_ENDPOINT,
            json={"query": query},
            timeout=30
        )
        if web_data.status_code != 200:
            return []
        
        raw_response = web_data.json()
        web_data = raw_response if isinstance(raw_response, list) else raw_response.get("results", [])
            
    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
    
    # Process and chunk results
    for item in web_data:
        if not item or not item.get('content'):
            continue
        
        # Split and format web content
        for chunk in text_splitter.split_text(item['content']):
            store_embedding(
                text=chunk,
                metadata={
                    "source": item['url']
                },
                content_type="web"
            )
            