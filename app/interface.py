import os
import streamlit as st
import time
from typing import List
from modules.vector_store.operations import cleanup_resources
from modules.processing import handle_file
from modules.vector_store.operations import store_embedding
from modules.qa_pipeline.generation import full_qa_pipeline
from modules.vector_store.setup import initialize_vector_store

USER_AVATAR = "https://img.icons8.com/?size=100&id=33100&format=png&color=000000"
BOT_AVATAR = "https://img.icons8.com/?size=100&id=102660&format=png&color=000000"


def display_welcome():
    """Interactive welcome message with expandable help"""
    with st.expander("🚀 **Welcome to Smart Doc Analyzer!**", expanded=True):
        st.markdown("""
        **How to use:**
        1. 📤 Upload files in sidebar
        2. ⚙️ Click 'Process Files'
        3. 💬 Chat with your data!
        
        **Supported Formats:**  
        - 📄 Documents
        - 🎤 Audio recordings  
        - 🖼️ Images  
        """)
        st.divider()
        st.caption("Tip: Use voice input with the 🎤 button!")

def file_processor(uploaded_files: List) -> None:
    """Robust file processing with enhanced error handling"""
    try:
        progress_bar = st.progress(0, text="Initializing processing...")
        total_files = len(uploaded_files)
        
        
        for idx, uploaded_file in enumerate(uploaded_files):
            file_path = f"{uploaded_file.name}"
            try:
                # Save file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process file
                items = handle_file(file_path)
                if not items:
                    raise ValueError(f"No processable content in {uploaded_file.name}")
                
                # Store embeddings
                for item in items:
                    store_embedding(
                        text=item["text"],
                        metadata=item["metadata"],
                        content_type=item["type"]
                    )
                
                # Track successful processing
                st.session_state.uploaded_files.append({
                    "name": uploaded_file.name,
                    "size": f"{uploaded_file.size//1024} KB",
                    "chunks": len(items),
                    "processed": True
                })
                
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                st.session_state.uploaded_files.append({
                    "name": uploaded_file.name,
                    "error": str(e)
                })
            finally:
                # Clean temp file immediately
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            # Update progress
            progress_bar.progress(
                (idx + 1) / total_files,
                text=f"Processed {uploaded_file.name} ({idx+1}/{total_files})"
            )
        
        progress_bar.empty()
        if total_files > 0:
            st.success(f"Successfully processed {total_files} files!")
            time.sleep(1)
            st.rerun()
            
    except Exception as e:
        st.error(f"Critical processing error: {str(e)}")
        cleanup_resources(perform_cleanup=True)
        st.rerun()

def file_status_sidebar():
    """Enhanced file status panel with file details"""
    with st.sidebar.expander("📁 **Uploaded Files**", expanded=True):
        if not st.session_state.uploaded_files:
            st.info("No Files processed yet")
            return
            
        for doc in st.session_state.uploaded_files:
            cols = st.columns([0.2, 1])
            status_icon = "✅" if doc.get("processed") else "❌"
            cols[0].subheader(status_icon)
            
            if "error" in doc:
                cols[1].error(f"{doc['name']}\n`{doc['error']}`")
            else:
                cols[1].caption(f"""
                **{doc['name']}**  
                Size: {doc.get('size', 'N/A')}  
                Chunks: {doc.get('chunks', 'N/A')}
                """)
                
def handle_streaming_response(response_stream):
    """Process real-time stream with <think> parsing"""
    reasoning_buffer = []
    answer_buffer = []
    in_reasoning = False
    
    # Create placeholders
    reasoning_placeholder = st.empty()
    answer_placeholder = st.empty()
    
    # Initial expander for reasoning
    with st.expander("🧠 Live Reasoning", expanded=True):
        reasoning_box = st.empty()
    
    # Process each chunk from stream
    for chunk in response_stream:
        if "<think>" in chunk:
            in_reasoning = True
            chunk = chunk.replace("<think>", "")
        elif "</think>" in chunk:
            in_reasoning = False
            chunk = chunk.replace("</think>", "")
        
        # Distribute content
        if in_reasoning:
            reasoning_buffer.append(chunk)
            with reasoning_box.container():
                st.markdown(f"""
                <style>
                    .reasoning {{
                        color: #666666;
                        font-style: italic;
                        font-size: 0.9em;
                    }}
                </style>
                {"".join(reasoning_buffer)}</div>
                """, unsafe_allow_html=True)
        else:
            answer_buffer.append(chunk)
            answer_placeholder.markdown("".join(answer_buffer))
    
    # Finalize session state storage
    return f"{''.join(answer_buffer)}"

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="Smart Doc Analyzer",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize vector store FIRST
    initialize_vector_store()
    
    # Session state initialization
    defaults = {
        "messages": [],
        "uploaded_files": [],
        "web_search": True,
        "dark_mode": False,
        "use_history": False
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)
    
    # Custom CSS for better spacing
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 350px;
    }
    .stChatInput {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Controls
    with st.sidebar:
        st.title("File Manager")
        
        # File Upload Section
        uploaded_files = st.file_uploader(
            "Upload Knowledge Files",
            accept_multiple_files=True
        )
        
        # Processing Control
        if uploaded_files:
            if st.button("⚡ Process Files", use_container_width=True):
                file_processor(uploaded_files)
        
        # File Status Panel
        file_status_sidebar()
        
        # System Controls
        with st.expander("⚙️ System Settings", expanded=False):
            
            if st.button("🔄 Soft Reset Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
                

            if st.button("🧹 Full System Reset", use_container_width=True):
                cleanup_resources(perform_cleanup=True)
                st.session_state.clear()
                st.success("System reset complete!")
                time.sleep(1)
                st.rerun()
    
    # Main Chat Interface
    st.title("💬 Smart File-Analyzer Chatbot")
    
    if not st.session_state.uploaded_files:
        display_welcome()
    else:
        # Chat History
        for msg in st.session_state.messages:
            avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
            with st.chat_message(msg["role"], avatar=avatar):  # 'user' or 'assistant'
                st.markdown(msg["content"])
                
        
        # Input Area
        input_col, web_col, history_col = st.columns([6, 2, 2], vertical_alignment="bottom")
            
        with input_col:
            prompt = st.chat_input("Ask anything about your files...")
        
        web_col.toggle(
            "🌐 Web Search",
            key="web_search",
            help="Include fresh web search results in knowledge base"
            )
                    
        history_col.checkbox(
                "🧠 Use chat history", 
                key="use_history",
                help="Include previous conversation context"
            )
        
        # Handle Query
        if prompt:
            # Add user message with avatar
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "avatar": USER_AVATAR 
            })
            
            # Display user message
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                try:
                    response_stream = full_qa_pipeline(
                        query=prompt,
                        use_web=st.session_state.web_search,
                        chat_history=st.session_state.messages if st.session_state.use_history else None
                    )
                    final_response = handle_streaming_response(response_stream)
                        
                    # Parse sources from response
                    sources = []
                    if "[Source:" in final_response:
                        source_parts = final_response.split("[Source:")[1:]
                        for part in source_parts:
                            source = part.split("]")[0].strip()
                            if source not in sources:
                                sources.append(source)
                    
                    
                    # Store message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_response,
                        "sources": sources,
                        "avatar": BOT_AVATAR  
                    })
                
                except Exception as e:
                    st.error(f"Failed to generate response: {str(e)}")

if __name__ == "__main__":
    main()
