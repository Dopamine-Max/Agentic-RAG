import os
from typing import List, Dict, Generator
from config import GROQ_API_KEY
from groq import Groq
from .retrieval import retrieve_context

def generate_response(query: str, context: List[Dict],
                     model_name: str = "deepseek-r1-distill-llama-70b",
                     chat_history: List[Dict] = None) -> Generator[str, None, None]:
    """
    Generates a natural language response using retrieved context and LLM
    
    Args:
        query: User's original question
            context: Retrieved context chunks
            model_name: model to use for generation
            chat_history: Chat history for LLM continuation
        
    Returns:
        Streamed response from LLM   
    """
    
    try:
        # Format context with sources
        context_str = "\n\n".join(
            [f"-> [Source: {chunk['metadata'].get('source', 'unknown')}]: "
            f"{chunk['text']}" 
            for chunk in context]
        )

        if not context_str:
            yield "No relevant information found. Please try another query."
            return

        client = Groq(api_key=GROQ_API_KEY)

        # Construct message chain
        messages = [{
            "role": "system",
            "content": (
                "You are an analytical assistant synthesizing information from provided sources. Ensure you follow the rules!\n"
                "Rules:\n"
                "1. Maintain RELEVANCE with USER QUERY \n"
                "2. Use ONLY provided Document Context\n"
                "3. NEVER invent beyond context\n"
                "4. ALWAYS CITE sources using [Source: ]\n"
                "5. Maintain CONVERSATION FLOW from Chat History\n"
            )
        }]

        if context_str:
            messages.append({
                "role": "system",
                "content": f"Document Context:\n{context_str}"
            })

        if chat_history:
            messages.append({"role": "system", "content": "Chat History:"})
            messages.extend([
                {"role": msg["role"], "content": msg["content"]}
                for msg in chat_history
            ])

        messages.append({"role": "user", "content": f"USER QUERY: {query}"})

        # Generate streamed response
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5,
            stream=True
        )
        
        # # TEMPORARY CONTEXT DISPLAY
        # import streamlit as st  # Add at top for production use
        # with st.expander("🔍 RAW CONTEXT (DEBUG)", expanded=False):
        #     st.markdown("### Retrieved Context Chunks")
        #     if context_str:
        #         st.markdown(messages)
        #     else:
        #         st.warning("No context available for this query")
        #     st.markdown("---")

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        print(f"Response generation failed: {str(e)}")
        yield "Error processing request"

def full_qa_pipeline(query: str, use_web: bool = True, 
                     chat_history: List[Dict] = None) -> str:
    """
    Complete Q&A pipeline from query to formatted response
    
    Args:
        query: Natural language question
        use_web: Whether to include web search results
        chat_history: Chat history for LLM continuation
        
    Returns:
        Formatted answer with citations
    """
    try:
        # Retrieve relevant context
        context = retrieve_context(query, use_web)
        
        # Generate final response
        return generate_response(query, context, chat_history=chat_history)
    
    except Exception as e:
        print(f"QA pipeline failed: {str(e)}")
        return "Unable to process your request at this time."
    
    
"""
Function to integrate Ollama LLM for response generation instead of Groq
"""    
# from langchain_ollama import ChatOllama
# from langchain.schema import HumanMessage, SystemMessage, AIMessage
# from config import OLLAMA_BASE_URL
# def generate_response(query: str, context: List[Dict], 
#                       model_name: str = "deepseek-r1:14b", chat_history: List[Dict] = None):
#     """
#     Generates a natural language response using retrieved context and LLM
    
#     Args:
#         query: User's original question
#         context: Retrieved context chunks
#         model_name: Ollama model to use for generation
#         chat_history: Chat history for LLM continuation
#     """
#     try:
#         # Format context with sources
#         if hasattr(context[0], 'payload'):  # Qdrant SearchResults
#             context_str = "\n\n".join(
#                 [f"-> [Source: {chunk.payload['metadata'].get('source', 'unknown')}]: "
#                  f"{chunk.payload['text']}" 
#                  for chunk in context]
#             )
#           else:  # Dictionary format
#               context_str = "\n\n".join(
#               [f"-> [Source: {chunk['metadata'].get('source', 'unknown')}]: "
#                f"{chunk['text']}" 
#                for chunk in context]
#     )

#         if not context_str:
#             return "No relevant information found. Please try another query."

#         # Initialize LLM with safety guardrails
#         llm = ChatOllama(
#             model=model_name,
#             base_url=OLLAMA_BASE_URL,
#             temperature=0.5,
#             system=(
#                 "You are an analytical assistant that synthesizes information from provided sources and conversation history. "
#                 "Follow these rules strictly:\n"
#                 "1. ANSWER USING ONLY PROVIDED CONTEXT \n "
#                 "2. NEVER INVENT ANYTHING BEYOND PROVIDED CONTEXT and HISTORY\n "
#                 "3. ALWAYS CITE SOURCES USING [Source: ] field in context\n "
#                 "4. ENSURE SEMANTIC RELEVANCE OF ANSWER\n "
#                 "Example: 'According to [Source Name] or (source_url)...'"
#                 "If you are unsure, say 'Based on my analysis of available information...'"
#             ), 
#             streaming=True
#         )
        
#         history_messages = []
#         if chat_history:
#             for msg in chat_history:
#                 if msg["role"] == "user":
#                     history_messages.append(HumanMessage(content=msg["content"]))
#                 elif msg["role"] == "assistant":
#                     history_messages.append(AIMessage(content=msg["content"]))

#         # Construct prompt
#         messages = [
#             SystemMessage(content=(
#                 "You are an analytical assistant that synthesizes information from provided sources and conversation history. " 
#                 "Follow these rules strictly:\n "
#                 "1. ANSWER USING ONLY PROVIDED CONTEXT (Context From Files:) \n "
#                 "2. NEVER INVENT ANYTHING BEYOND PROVIDED CONTEXT and HISTORY\n "
#                 "3. ALWAYS CITE SOURCES USING [Source:] field in context\n "
#                 "4. ENSURE SEMANTIC RELEVANCE OF ANSWER\n "
#                 "Example: 'According to [Source Name] or (source_url)...' "
#                 "If you are unsure, say 'Based on my analysis of available information...'")),
#             SystemMessage(content="Current Document Context:"),
#             SystemMessage(content=f"Context from Files:\n{context_str}"),
#             HumanMessage(content=f"Message History:\n"),
#             *history_messages,
#             HumanMessage(content=f"New Question: {query}")
#         ]
        
#         # TEMPORARY CONTEXT DISPLAY
#         import streamlit as st  # Add at top for production use
#         with st.expander("🔍 RAW CONTEXT (DEBUG)", expanded=False):
#             st.markdown("### Retrieved Context Chunks")
#             if context_str:
#                 st.markdown(messages)
#             else:
#                 st.warning("No context available for this query")
#             st.markdown("---")

#         # Generate and return response
#         for chunk in llm.stream(messages):
#             yield chunk.content

#     except Exception as e:
#         print(f"Response generation failed: {str(e)}")
#         return "I encountered an error processing your request."