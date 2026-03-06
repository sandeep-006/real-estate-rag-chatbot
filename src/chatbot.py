import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

BASE_DIR        = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def load_vectorstore():
    """Load the saved FAISS vector store."""
    print("📦 Loading vector store...")

    if not VECTORSTORE_DIR.exists():
        print("❌ Vector store not found! Run ingest.py first.")
        sys.exit(1)

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

    print("✅ Vector store loaded successfully.")
    return vectorstore


def create_llm():
    """Initialize Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file!")
        sys.exit(1)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024
    )
    print("✅ Groq LLM initialized.")
    return llm


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(chat_history):
    """Format chat history into readable string."""
    if not chat_history:
        return "No previous conversation."
    lines = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


def create_chain(vectorstore, llm):
    """Build RAG chain manually using LangChain core."""
    print("🔗 Building RAG chain...")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

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
- Keep answers concise but complete

Context from knowledge base:
{context}

Conversation history:
{chat_history}"""),
        ("human", "{question}")
    ])

    # Build chain manually using LCEL (LangChain Expression Language)
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

    print("✅ RAG chain ready.")
    return chain, retriever


def get_answer(chain, retriever, question, chat_history):
    """Send a question and get an answer."""
    answer = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    # Fetch source docs separately for display
    sources = retriever.invoke(question)
    return answer, sources


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  🏠 Real Estate Chatbot — Chain Test")
    print("=" * 50)

    vectorstore         = load_vectorstore()
    llm                 = create_llm()
    chain, retriever    = create_chain(vectorstore, llm)

    chat_history = []

    test_questions = [
        "What is EMD in real estate?",
        "Tell me about properties available in Austin Texas",
        "What are closing costs?"
    ]

    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)
        answer, sources = get_answer(chain, retriever, question, chat_history)
        print(f"💬 Answer: {answer}")
        print(f"📚 Sources used: {len(sources)} chunk(s)")
        print("-" * 40)

        # Update chat history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))