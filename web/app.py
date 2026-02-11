"""
Streamlit web interface for the RAG system.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_system import Config, RAGPipeline
from src.rag_system.utils.logger import get_logger

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline(config_path=None):
    """Load and cache the RAG pipeline."""
    if config_path and Path(config_path).exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    
    return RAGPipeline(config)


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Advanced RAG System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        config_file = st.text_input(
            "Config File (optional)",
            placeholder="configs/default.yaml"
        )
        
        if st.button("🔄 Reload Pipeline"):
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        # Retrieval settings
        st.subheader("Retrieval Settings")
        top_k = st.slider("Top K Results", 1, 20, 5)
        enable_streaming = st.checkbox("Enable Streaming", value=True)
        
        st.divider()
        
        # About
        st.subheader("About")
        st.info("""
        This is a production-grade RAG system with:
        - 🔍 Hybrid Search (Dense + Sparse)
        - 🎯 Re-ranking
        - 💬 Conversation Memory
        - 🚀 Local LLM Support
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Ingest Documents", "📊 Statistics"])
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = "web_session"
    
    # Chat tab
    with tab1:
        st.header("Ask Questions")
        
        # Load pipeline
        try:
            pipeline = load_pipeline(config_file if config_file else None)
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            return
        
        # Check if documents are loaded
        stats = pipeline.get_stats()
        if stats['total_chunks'] == 0:
            st.warning("⚠️ No documents found. Please ingest documents first in the 'Ingest Documents' tab.")
        else:
            st.success(f"✅ {stats['total_chunks']} chunks loaded from {stats['vector_store_chunks']} documents")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message["sources"][:3], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i}</strong> (Score: {source['score']:.3f})<br>
                                <em>{source['source']}</em><br>
                                <small>{source['content'][:200]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                if enable_streaming:
                    # Streaming response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in pipeline.stream_query(
                        prompt,
                        top_k=top_k,
                        conversation_id=st.session_state.conversation_id
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    # Get sources (can't get from stream, so query again)
                    # In production, you'd want to refactor this
                    sources_data = []
                else:
                    # Non-streaming response
                    with st.spinner("Thinking..."):
                        response = pipeline.query(
                            prompt,
                            top_k=top_k,
                            conversation_id=st.session_state.conversation_id
                        )
                    
                    full_response = response.answer
                    st.markdown(full_response)
                    
                    # Prepare sources
                    sources_data = [
                        {
                            "score": s.score,
                            "source": s.chunk.metadata.get('source', 'Unknown'),
                            "content": s.chunk.content
                        }
                        for s in response.sources[:3]
                    ]
                    
                    # Show sources
                    if sources_data:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources_data, 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i}</strong> (Score: {source['score']:.3f})<br>
                                    <em>{source['source']}</em><br>
                                    <small>{source['content'][:200]}...</small>
                                </div>
                                """, unsafe_allow_html=True)
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_data if not enable_streaming else []
            })
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
    
    # Ingest tab
    with tab2:
        st.header("📄 Ingest Documents")
        
        pipeline = load_pipeline(config_file if config_file else None)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload
            st.subheader("Upload Files")
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'md', 'docx', 'html']
            )
            
            if st.button("📤 Ingest Uploaded Files") and uploaded_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_chunks = 0
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save temporarily
                    temp_path = Path("temp") / uploaded_file.name
                    temp_path.parent.mkdir(exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Ingest
                    try:
                        num_chunks = pipeline.ingest_file(str(temp_path))
                        total_chunks += num_chunks
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    finally:
                        # Clean up
                        temp_path.unlink()
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Ingested {total_chunks} chunks from {len(uploaded_files)} files!")
                st.cache_resource.clear()
                st.rerun()
        
        with col2:
            # Directory path
            st.subheader("Ingest Directory")
            dir_path = st.text_input("Directory Path")
            recursive = st.checkbox("Recursive", value=True)
            
            if st.button("📁 Ingest Directory") and dir_path:
                if Path(dir_path).exists():
                    with st.spinner("Ingesting documents..."):
                        try:
                            num_chunks = pipeline.ingest_directory(
                                dir_path,
                                recursive=recursive
                            )
                            st.success(f"✅ Ingested {num_chunks} chunks!")
                            st.cache_resource.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.error("Directory not found!")
    
    # Statistics tab
    with tab3:
        st.header("📊 System Statistics")
        
        pipeline = load_pipeline(config_file if config_file else None)
        stats = pipeline.get_stats()
        
        # Display stats in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Chunks", stats['total_chunks'])
            st.metric("Embedding Dimension", stats['embedding_dimension'])
        
        with col2:
            st.metric("Chunking Strategy", stats['chunking_strategy'])
            st.metric("Retrieval Strategy", stats['retrieval_strategy'])
        
        with col3:
            st.metric("LLM Provider", stats['llm_provider'])
            st.metric("LLM Model", stats['llm_model'])
        
        # Configuration details
        st.subheader("Configuration Details")
        st.json(stats)
        
        # Clear collection
        st.divider()
        st.subheader("⚠️ Danger Zone")
        if st.button("🗑️ Clear All Documents", type="secondary"):
            confirm = st.checkbox("I understand this will delete all documents")
            if confirm and st.button("Confirm Delete"):
                pipeline.clear_collection()
                st.success("Collection cleared!")
                st.cache_resource.clear()
                st.rerun()


def create_app(config):
    """Create FastAPI app for production deployment."""
    # This is for production API deployment
    # For now, we use Streamlit
    pass


if __name__ == "__main__":
    main()
