from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma

def create_vector_store(chunks):
    # Use an efficient embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.persist()
    return vector_db
