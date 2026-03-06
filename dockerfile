# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Run ingestion pipeline first to build vectorstore
RUN python src/ingest.py

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit on port 8080 (Cloud Run requires 8080)
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

---

### File 2: `.dockerignore`
Create `.dockerignore` in root folder:
```
venv/
__pycache__/
*.pyc
*.pyo
.env
.git
.gitignore
vectorstore/
```

> ⚠️ Notice we exclude `vectorstore/` — it gets **rebuilt inside Docker** using `ingest.py`

---

### File 3: Update `requirements.txt`
Replace everything in `requirements.txt` with this:
```
langchain==1.2.10
langchain-core==1.2.17
langchain-community==0.4.1
langchain-groq==1.1.2
langchain-huggingface
langchain-classic
faiss-cpu
pypdf
streamlit
python-dotenv
tiktoken
sentence-transformers
torch --index-url https://download.pytorch.org/whl/cpu
```

---

### File 4: Update `.gitignore`
Replace everything in `.gitignore` with:
```
venv/
vectorstore/
.env
__pycache__/
*.pyc
.DS_Store
*.egg-info/
.dockerignore