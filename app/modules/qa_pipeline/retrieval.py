from typing import List, Dict, Any
import streamlit as st
from modules.vector_store.operations import retrieve_from_qdrant
from modules.processing.web_processor import process_web_search

def retrieve_context(query: str, use_web: bool = True) -> List[Dict[str, Any]]:
    """
    Unified retrieval pipeline combining vector store and web results.
    
    Args:
        query: User's natural language query
        use_web: Whether to include fresh web results
        
    Returns:
        Combined and prioritized context for LLM
    """
    if use_web:
        # Trigger web search but don't use its return value
        with st.status("🌐 Augmenting knowledge base...", expanded=False):
            process_web_search(query)
    
    return retrieve_from_qdrant(query, top_k=10)