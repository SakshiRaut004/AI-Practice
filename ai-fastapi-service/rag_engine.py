import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Initialize persistent ChromaDB store locally
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use a lightweight sentence-transformer model for generating local embeddings
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="enterprise_docs",
    embedding_function=default_ef
)

def process_pdf_and_ingest(file_path: str, doc_id: str):
    """
    Extracts text from a PDF, applies recursive chunking, 
    and indexes vector embeddings into ChromaDB.
    """
    reader = PdfReader(file_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {page_num + 1} ---\n" + text

    # Recursive Character Text Splitting with overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    # Ingest chunks into ChromaDB
    documents = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({"doc_id": doc_id, "chunk_index": idx})
        ids.append(f"{doc_id}_chunk_{idx}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    return {"status": "success", "total_chunks": len(chunks)}


def query_knowledge_base(query_text: str, top_k: int = 3):
    """
    Performs vector similarity search against ChromaDB.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )
    
    retrieved_contexts = results["documents"][0] if results["documents"] else []
    return retrieved_contexts