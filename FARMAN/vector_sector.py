from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma

def create_vector_store(chunks):
    # Use an efficient embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        chunks,  # NOT chunks
        embedding_model,
        persist_directory="./chroma_db_hf",
        client_settings={"database_impl": "duckdb"})  # Optional fallback for sqlite3
    vector_db.persist()
    return vector_db
