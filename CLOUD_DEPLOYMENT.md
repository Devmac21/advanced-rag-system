# Cloud Deployment Guide

## ☁️ Deploy Your RAG System to the Cloud

This guide covers multiple cloud deployment options from **FREE** to production-grade.

---

## 🆓 Option 1: Hugging Face Spaces (RECOMMENDED - FREE!)

**Perfect for: Portfolio demos, sharing with recruiters**

### Setup Steps:

1. **Create Hugging Face Account**
   - Go to https://huggingface.co
   - Sign up (free)

2. **Create New Space**
   ```
   - Click "New Space"
   - Name: "advanced-rag-system"
   - SDK: Streamlit
   - Hardware: CPU basic (free)
   ```

3. **Prepare Files**
   Create `app.py` in project root:
   ```python
   # This will be your main Streamlit app for HF Spaces
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent))
   
   from web.app import main
   
   if __name__ == "__main__":
       main()
   ```

4. **Create `requirements-hf.txt`** (lighter for cloud):
   ```
   streamlit==1.28.0
   sentence-transformers==2.2.0
   faiss-cpu==1.7.4
   pypdf==3.15.0
   python-docx==1.0.0
   httpx==0.24.0
   groq  # For LLM API
   ```

5. **Push to HF Space**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/advanced-rag-system
   git push hf main
   ```

6. **Configure API Keys** (in HF Space settings):
   - Add secret: `GROQ_API_KEY=your_key_here`

**Result:** Your RAG system at `https://huggingface.co/spaces/YOUR_USERNAME/advanced-rag-system`

---

## 🎨 Option 2: Streamlit Community Cloud (FREE!)

**Perfect for: Quick demos, prototypes**

### Setup Steps:

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/advanced-rag.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your GitHub repo
   - Main file: `web/app.py`
   - Click "Deploy"

3. **Add Secrets** (in Streamlit Cloud dashboard):
   ```toml
   # .streamlit/secrets.toml
   GROQ_API_KEY = "your_key_here"
   ```

**Result:** Your app at `https://YOUR_APP_NAME.streamlit.app`

**Limitations:**
- 1GB RAM (use smaller models)
- Sleeps after inactivity
- Shared CPU

---

## 🚂 Option 3: Railway.app ($5/month)

**Perfect for: Development/staging environment**

### Setup Steps:

1. **Sign up** at https://railway.app

2. **Create New Project**
   - "Deploy from GitHub repo"
   - Select your repo

3. **Configure**
   ```
   Start Command: streamlit run web/app.py --server.port $PORT
   ```

4. **Add Environment Variables**:
   ```
   GROQ_API_KEY=your_key
   RAG_LLM__PROVIDER=groq
   ```

5. **Deploy**
   - Automatic on every git push

**Cost:** $5/month (512MB RAM)

---

## 🎯 Option 4: Using Groq API (FREE Tier)

**Perfect for: Fast, free LLM inference**

### Why Groq?
- ✅ **FREE tier:** 14,400 requests/day
- ✅ **Super fast:** 500+ tokens/second
- ✅ **Good models:** Llama 3, Mixtral, Gemma
- ✅ **No credit card** needed for free tier

### Setup:

1. **Get API Key**
   ```
   - Go to https://console.groq.com
   - Sign up (free)
   - Create API key
   ```

2. **Install Groq SDK**
   ```bash
   pip install groq
   ```

3. **Create Groq LLM Implementation**
   
   I'll create this for you:
