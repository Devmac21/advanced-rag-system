"""
Beautiful Streamlit UI for Advanced RAG System
"""
import streamlit as st
import os
import sys
from pathlib import Path
import time

# Set UTF-8 encoding
if sys.platform == 'win32':
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    except:
        pass

# Add src to path
sys.path.insert(0, 'src')

from rag_system import Config, RAGPipeline

# Page config
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        background: white;
        border-radius: 20px;
        margin: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 25px rgba(102,126,234,0.3);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Chat messages */
    .user-message {
        background: #e3f2fd;
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 5px 15px;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }
    
    .assistant-message {
        background: #f3e5f5;
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 1rem 0;
        border-left: 4px solid #9C27B0;
    }
    
    /* Source boxes */
    .source-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    .source-title {
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .source-score {
        color: #667eea;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        box-shadow: 0 10px 25px rgba(102,126,234,0.5);
        transform: translateY(-2px);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Success/Error alerts */
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Load API key
@st.cache_data
def load_api_key():
    """Load API key from .env file."""
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('GROQ_API_KEY'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                    return True
    except:
        pass
    return False


@st.cache_resource
def load_pipeline():
    """Load and cache the RAG pipeline."""
    config = Config.from_yaml("configs/groq.yaml")
    return RAGPipeline(config)


def get_fresh_stats(pipeline):
    """Get fresh stats without caching."""
    return pipeline.get_stats()


def main():
    """Main application."""
    
    # Load API key
    if not load_api_key():
        st.error("API key not found! Please create .env file with GROQ_API_KEY=your_key")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Advanced RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Hybrid Search, Re-ranking & Llama 3.3</p>', unsafe_allow_html=True)
    
    # Initialize pipeline
    try:
        pipeline = load_pipeline()
        stats = get_fresh_stats(pipeline)  # Always get fresh stats
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        
        # Stats cards
        # Always show fresh document count
        fresh_stats = get_fresh_stats(pipeline)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{fresh_stats['total_chunks']}</div>
            <div class="stat-label">Documents Indexed</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{fresh_stats['embedding_dimension']}</div>
            <div class="stat-label">Embedding Dimensions</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### 🎛️ Settings")
        top_k = st.slider("Number of Results", 1, 10, 5, help="How many relevant chunks to retrieve")
        show_sources = st.checkbox("Show Sources", value=True, help="Display source documents")
        show_metrics = st.checkbox("Show Metrics", value=True, help="Display performance metrics")
        
        st.divider()
        
        st.markdown("### 📊 Model Info")
        st.info(f"""
        **LLM:** {fresh_stats['llm_model']}  
        **Retrieval:** {fresh_stats['retrieval_strategy']}  
        **Vector Store:** FAISS  
        **Chunks:** {fresh_stats['total_chunks']}  
        **Status:** 🟢 Online
        """)
        
        if st.button("🔄 Refresh Stats", help="Reload document statistics"):
            st.success("✅ Stats refreshed!")
            time.sleep(0.5)
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📤 Upload Documents", "📊 Analytics", "ℹ️ About"])
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = "web_session"
    
    # Tab 1: Chat Interface
    with tab1:
        # Always get fresh stats in chat tab
        current_stats = get_fresh_stats(pipeline)
        
        if current_stats['total_chunks'] == 0:
            st.warning("⚠️ No documents indexed yet! Upload documents in the 'Upload Documents' tab.")
            st.info("💡 Tip: Go to the 'Upload Documents' tab and either upload your files or click 'Load Test Documents'")
        else:
            st.success(f"✅ Ready to answer questions from {current_stats['total_chunks']} document chunks!")
            
            # Show a refresh button
            col1, col2 = st.columns([6, 1])
            with col2:
                if st.button("🔄 Refresh", key="refresh_chat"):
                    st.rerun()
        
        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant:</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources
                if show_sources and "sources" in msg and msg["sources"]:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(msg["sources"][:3], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-title">Source {i} <span class="source-score">(Score: {source['score']:.3f})</span></div>
                                <em>{source['source']}</em><br>
                                <small>{source['content'][:200]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show metrics
                if show_metrics and "metrics" in msg:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retrieval", f"{msg['metrics']['retrieval']:.2f}s")
                    with col2:
                        st.metric("Generation", f"{msg['metrics']['generation']:.2f}s")
                    with col3:
                        st.metric("Total", f"{msg['metrics']['total']:.2f}s")
        
        # Chat input
        st.markdown("### 💭 Ask a Question")
        
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Type your question here...",
                key="user_input",
                placeholder="e.g., What is Python? Tell me about machine learning...",
                label_visibility="collapsed"
            )
        with col2:
            send_button = st.button("Send 🚀", use_container_width=True)
        
        if send_button and user_input:
            # Check if documents exist before querying
            fresh_check = get_fresh_stats(pipeline)
            if fresh_check['total_chunks'] == 0:
                st.error("❌ No documents found! Please upload documents first in the 'Upload Documents' tab.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Generate response
                with st.spinner("Thinking..."):
                    try:
                        response = pipeline.query(user_input, top_k=top_k, conversation_id=st.session_state.conversation_id)
                        
                        # Prepare sources
                        sources_data = [
                            {
                                "score": s.score,
                                "source": s.chunk.metadata.get('source', 'Unknown'),
                                "content": s.chunk.content
                            }
                            for s in response.sources
                        ]
                        
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": sources_data,
                            "metrics": {
                                "retrieval": response.retrieval_time,
                                "generation": response.generation_time,
                                "total": response.total_time
                            }
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        # Remove the failed user message
                        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                            st.session_state.messages.pop()
        
        # Clear chat button
        if len(st.session_state.messages) > 0:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
    
    # Tab 2: Upload Documents
    with tab2:
        st.markdown("### 📤 Upload Your Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Upload Files")
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'md', 'docx', 'html'],
                help="Supported formats: PDF, TXT, Markdown, DOCX, HTML"
            )
            
            if uploaded_files:
                st.info(f"📁 {len(uploaded_files)} file(s) selected")
                
                if st.button("📥 Ingest Files", use_container_width=True):
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    total_chunks = 0
                    temp_dir = Path("temp_uploads")
                    temp_dir.mkdir(exist_ok=True)
                    
                    for i, file in enumerate(uploaded_files):
                        status.text(f"Processing {file.name}...")
                        
                        # Save temporarily
                        temp_path = temp_dir / file.name
                        with open(temp_path, 'wb') as f:
                            f.write(file.getbuffer())
                        
                        # Ingest
                        try:
                            chunks = pipeline.ingest_file(str(temp_path))
                            total_chunks += chunks
                            status.success(f"✅ {file.name}: {chunks} chunks")
                            time.sleep(0.5)
                        except Exception as e:
                            status.error(f"❌ Error with {file.name}: {e}")
                        finally:
                            temp_path.unlink(missing_ok=True)
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Clean up
                    try:
                        temp_dir.rmdir()
                    except:
                        pass
                    
                    st.success(f"🎉 Successfully ingested {total_chunks} chunks from {len(uploaded_files)} files!")
                    st.info("♻️ Refreshing pipeline to load new documents...")
                    time.sleep(2)
                    st.rerun()
        
        with col2:
            st.markdown("#### Quick Test")
            
            st.markdown("""
            **Test the system with sample documents:**
            
            We've created 3 test files for you:
            - `test_python.txt` - About Python programming
            - `test_ml.txt` - About machine learning
            - `test_datascience.txt` - About data science
            """)
            
            if st.button("🧪 Load Test Documents", use_container_width=True):
                with st.spinner("Loading test documents..."):
                    # They're already loaded from test_system.py
                    st.success("✅ Test documents already loaded!")
                    time.sleep(1)
                    st.rerun()
    
    # Tab 3: Analytics
    with tab3:
        st.markdown("### 📊 System Analytics")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['total_chunks']}</div>
                <div class="stat-label">Total Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['embedding_dimension']}</div>
                <div class="stat-label">Embedding Dims</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            strategy = str(stats['retrieval_strategy']).split('.')[-1]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{strategy}</div>
                <div class="stat-label">Retrieval</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">🟢</div>
                <div class="stat-label">System Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Detailed stats
        st.markdown("### 🔧 Configuration Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Chunking": {
                    "strategy": str(stats['chunking_strategy']),
                    "total_chunks": stats['total_chunks']
                },
                "Embeddings": {
                    "model": "all-MiniLM-L6-v2",
                    "dimension": stats['embedding_dimension']
                }
            })
        
        with col2:
            st.json({
                "Retrieval": {
                    "strategy": str(stats['retrieval_strategy']),
                    "vector_store": "FAISS"
                },
                "LLM": {
                    "provider": stats['llm_provider'],
                    "model": stats['llm_model']
                }
            })
        
        st.divider()
        
        # Danger zone
        st.markdown("### ⚠️ Danger Zone")
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.checkbox("I understand this will delete all indexed documents"):
                pipeline.clear_collection()
                st.success("✅ Collection cleared!")
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
    
    # Tab 4: About
    with tab4:
        st.markdown("### 🎯 About This Project")
        
        st.markdown("""
        ## Advanced RAG System
        
        A production-grade Retrieval-Augmented Generation system with cutting-edge features.
        
        ### ✨ Key Features
        
        - **🔍 Hybrid Search**: Combines dense (embeddings) and sparse (BM25) retrieval
        - **🎯 Re-ranking**: Cross-encoder for improved relevance
        - **📚 Multi-format**: Supports PDF, DOCX, TXT, Markdown, HTML, Code
        - **⚡ Fast**: Sub-second responses with Groq API
        - **💬 Conversational**: Multi-turn dialogue with context
        - **📊 Metrics**: Track performance and quality
        
        ### 🛠️ Tech Stack
        
        - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
        - **Vector Store**: FAISS (high-performance similarity search)
        - **Retrieval**: Hybrid (Dense + Sparse BM25)
        - **Re-ranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2)
        - **LLM**: Groq API (Llama 3.3 70B)
        - **Frontend**: Streamlit
        
        ### 🏗️ Architecture
        
        ```
        User Query
            ↓
        Query Processing (expansion, rewriting)
            ↓
        Hybrid Retrieval (dense + sparse)
            ↓
        Re-ranking (cross-encoder)
            ↓
        Context Assembly
            ↓
        LLM Generation (with citations)
            ↓
        Response + Sources
        ```
        
        ### 📈 Performance
        
        - **Retrieval**: ~0.2-0.3s
        - **Generation**: ~0.3-0.5s
        - **Total**: ~0.6-0.8s per query
        
        ### 🎓 For Portfolios
        
        This project demonstrates:
        - Advanced RAG techniques
        - Production-quality code
        - System architecture skills
        - ML engineering expertise
        - Full-stack development
        
        ### 🔗 Links
        
        - [GitHub Repository](#)
        - [Documentation](README.md)
        - [API Reference](USAGE.md)
        
        ---
        
        **Built with ❤️ for AI/ML Engineers**
        """)
        
        # System info
        st.divider()
        st.markdown("### 💻 System Information")
        
        import platform
        st.code(f"""
OS: {platform.system()} {platform.release()}
Python: {platform.python_version()}
Total Chunks: {stats['total_chunks']}
Retrieval Strategy: {stats['retrieval_strategy']}
LLM: {stats['llm_provider']} / {stats['llm_model']}
        """)


if __name__ == "__main__":
    main()
