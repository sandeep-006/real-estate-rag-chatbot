# 🏠 Real Estate RAG Chatbot

A conversational AI chatbot for real estate queries built with:
- 🧠 Groq LLM (llama-3.3-70b-versatile)
- 📚 RAG with FAISS vector store
- 🔗 LangChain
- 🎨 Streamlit UI

## Setup

1. Clone the repo
2. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Add your Groq API key to .env:
   GROQ_API_KEY=your_key_here

5. Run ingestion pipeline:
   python src/ingest.py

6. Launch the chatbot:
   streamlit run src/app.py

## Project Structure
CHATBOT_AI/
├── data/                  # Knowledge base documents
├── src/
│   ├── ingest.py          # Document ingestion pipeline
│   ├── chatbot.py         # RAG chain logic
│   └── app.py             # Streamlit UI
├── vectorstore/           # FAISS index (auto-generated)
├── .env                   # API keys (never commit this)
└── requirements.txt
```

---

## ✅ Final Project Structure

Your complete project should look like this:
```
CHATBOT_AI/
│
├── data/
│   ├── real_estate_faq.txt        ✅
│   ├── property_listings.txt      ✅
│   └── real_estate_terms.txt      ✅
│
├── src/
│   ├── ingest.py                  ✅
│   ├── chatbot.py                 ✅
│   └── app.py                     ✅
│
├── vectorstore/
│   ├── index.faiss                ✅
│   └── index.pkl                  ✅
│
├── venv/                          ✅
├── .env                           ✅
├── .gitignore                     ✅
├── README.md                      ✅
└── requirements.txt               ✅