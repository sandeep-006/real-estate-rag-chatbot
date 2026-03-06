import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load .env (works locally, ignored on cloud)
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def load_documents():
    """Load all .txt files from the data/ folder."""
    print("📂 Loading documents from data/ folder...")

    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )

    documents = loader.load()

    if not documents:
        print("❌ No documents found! Make sure .txt files are inside data/ folder.")
        sys.exit(1)

    print(f"✅ Loaded {len(documents)} document(s) successfully.")
    return documents


def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    print("\n✂️  Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks from your documents.")
    return chunks


def create_vectorstore(chunks):
    """Convert chunks to embeddings using HuggingFace (free & local)."""
    print("\n🔢 Creating embeddings using HuggingFace (free, runs locally)...")
    print("   ⏳ First time will download the model (~90MB). Please wait...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save locally
    VECTORSTORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"✅ Vector store saved to: {VECTORSTORE_DIR}")
    return vectorstore


def main():
    print("=" * 50)
    print("  🏠 Real Estate Chatbot — Ingestion Pipeline")
    print("=" * 50)

    documents = load_documents()
    chunks    = split_documents(documents)
    create_vectorstore(chunks)

    print("\n🎉 Ingestion complete! Ready for chatbot.")
    print("=" * 50)


if __name__ == "__main__":
    main()