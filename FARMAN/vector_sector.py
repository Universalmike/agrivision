from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def create_vector_store(docs):
    # Use an efficient embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(split_docs, embedding_model, persist_directory="./chroma_db_hf")
    vector_db.persist()
    return vector_db
