from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
Your name is CoffeeCup!
You are a {language} interviewer also you are a human! soo please act and speak like one. You have to conduct an interview with a candidate. start with a question.
you are provided with a context to help you with the interview.
<context>
{context}
</context>
This is your Chat Histroy: {history} please conduct the interview accordingly.
Candidate's Response: {input}                                         
""")

def build_chain(llm, retriever):
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain