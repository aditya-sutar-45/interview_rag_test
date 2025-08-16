import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

VECTOR_STORE_PATH = "./vector_store/db/faiss_store"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def create_or_load_vector_store(documents, force_recreate=False):
    if os.path.exists(VECTOR_STORE_PATH) and not force_recreate:
        print("Loading existing FAISS vector store...")
        try: 
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, get_embeddings(), allow_dangerous_deserialization=True)
            test_doc = vector_store.similarity_search("test", k=1, filter={"language": "java"})
            print(f"Test document found: {len(test_doc)} results.")

            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Recreating vector store due to error.")
            force_recreate = True

    if force_recreate or not os.path.exists(VECTOR_STORE_PATH):
        print("Creating new FAISS vector store...")

        if os.path.exists(VECTOR_STORE_PATH):
            import shutil
            shutil.rmtree(VECTOR_STORE_PATH)

        vector_store = FAISS.from_documents(documents, get_embeddings())

        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)

        print(f"Vector store created and saved at {VECTOR_STORE_PATH}")

        test_doc = vector_store.similarity_search("test", k=1, filter={"language": "java"})
        print(f"Test document found: {len(test_doc)} results.")
        return vector_store

def get_retriever(vector_store, filter=None, k=5):
    if filter:
        return vector_store.as_retriever(search_kwargs={"k": k, "filter": {"language": filter}})
    else:
        search_kwargs = {"k": k}
        return vector_store.as_retriever(search_kwargs=search_kwargs)