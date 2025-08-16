from loaders.loader import load_all_md_file
from dotenv import load_dotenv
import os
from vector_store.faiss_store import create_or_load_vector_store, get_retriever
from chains.interview_chain import build_chain
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

documents = load_all_md_file()
print(f"Loaded {len(documents)} documents.")

languages = set()
for doc in documents:
    if 'language' in doc.metadata:
        languages.add(doc.metadata['language'])
print(f"Available languages: {', '.join(languages)}")

vector_store = create_or_load_vector_store(documents, force_recreate=False)

try:
    java_docs = vector_store.similarity_search("hello interview", k=5, filter={"language": "javascript"})
    print(f"Found {len(java_docs)} JavaScript documents.")
except Exception as e:
    print(f"Error during similarity search: {e}")
retriever = get_retriever(vector_store, filter="javascript", k=5)

llm = GoogleGenerativeAI(model="gemini-2.5-flash")
retrieval_chain = build_chain(llm, retriever)

interview_response = retrieval_chain.invoke({"input": "hello, lets start the interview", "history": [], "language": "javascript"})

print(interview_response)
print(interview_response['context'])