import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# ── Auto-build vectorstore if missing (needed for Streamlit Cloud) ─────────────
if not VECTORSTORE_DIR.exists() or not (VECTORSTORE_DIR / "index.faiss").exists():
    st.set_page_config(page_title="🏠 Real Estate Chatbot", page_icon="🏠")
    st.info("⚙️ Building knowledge base for first time... Please wait 2-3 minutes.")
    with st.spinner("Building vectorstore..."):
        sys.path.append(str(Path(__file__).resolve().parent))
        from ingest import main as run_ingest
        run_ingest()
    st.success("✅ Knowledge base built! Reloading...")
    st.rerun()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 Real Estate Chatbot",
    page_icon="🏠",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .st-emotion-cache-1c7y2kd { background-color: #e8f4f8; }
    .title-container {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }
    .title-text {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .subtitle-text {
        color: #a0c4ff;
        font-size: 1rem;
        margin: 5px 0 0 0;
    }
    .stats-box {
        background: white;
        border-left: 4px solid #1a1a2e;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 15px;
        font-size: 0.85rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Resources (cached so it only runs once) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    """Load vectorstore and LLM — runs only once."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return llm, retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(chat_history):
    if not chat_history:
        return "No previous conversation."
    lines = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


def build_chain(llm, retriever, chat_history):
    """Build the RAG chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert real estate assistant with deep knowledge 
about properties, buying/selling process, legal terms, and market trends.

Use the context below from our real estate knowledge base to answer accurately.

Guidelines:
- Be friendly, clear and professional
- Use the context to answer directly
- If you don't know, say: "I don't have that information right now, 
  please consult a licensed real estate agent."
- Never make up property prices or legal advice
- Use bullet points or formatting where it helps readability
- Keep answers concise but complete

Context from knowledge base:
{context}

Conversation history:
{chat_history}"""),
        ("human", "{question}")
    ])

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(
                retriever.invoke(x["question"])
            )),
            chat_history=RunnableLambda(lambda x: format_history(
                x.get("chat_history", [])
            ))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── UI ─────────────────────────────────────────────────────────────────────────
def main():

    # Header
    st.markdown("""
    <div class="title-container">
        <p class="title-text">🏠 Real Estate Assistant</p>
        <p class="subtitle-text">Ask me anything about properties, buying, selling, or real estate terms</p>
    </div>
    """, unsafe_allow_html=True)

    # Load resources
    with st.spinner("🔄 Loading AI model... please wait"):
        try:
            llm, retriever = load_resources()
        except Exception as e:
            st.error(f"❌ Failed to load resources: {e}")
            st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("## 💡 Sample Questions")
        sample_questions = [
            "What is EMD in real estate?",
            "Show me properties in Austin Texas",
            "What are closing costs?",
            "What is the difference between pre-qualification and pre-approval?",
            "What is an HOA?",
            "Explain LTV ratio",
            "What is a contingency?",
            "Tell me about the Lakefront Retreat property",
        ]

        for q in sample_questions:
            if st.button(q, use_container_width=True):
                st.session_state.selected_question = q

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Chat Stats")
        msg_count = len(st.session_state.get("messages", []))
        st.markdown(f"**Messages:** {msg_count}")
        st.markdown("**Model:** llama-3.3-70b")
        st.markdown("**DB:** FAISS (local)")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Welcome message
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("""👋 **Welcome! I'm your Real Estate Assistant.**

I can help you with:
- 🏡 **Property listings** in Austin, Texas
- 📋 **Buying & selling** process questions
- 📖 **Real estate terms** and definitions
- 💰 **Financing** concepts like EMD, closing costs, LTV

**Go ahead and ask me anything!**""")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle sample question click from sidebar
    if "selected_question" in st.session_state:
        user_input = st.session_state.selected_question
        del st.session_state.selected_question
    else:
        user_input = st.chat_input("Ask about properties, prices, legal terms...")

    # Process input
    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    chain = build_chain(llm, retriever, st.session_state.chat_history)
                    answer = chain.invoke({
                        "question": user_input,
                        "chat_history": st.session_state.chat_history
                    })

                    st.markdown(answer)

                    # Update histories
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    st.session_state.chat_history.append(
                        HumanMessage(content=user_input)
                    )
                    st.session_state.chat_history.append(
                        AIMessage(content=answer)
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()