import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langserve import add_routes
from fastapi import FastAPI
import uvicorn
import time

from dotenv import load_dotenv
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Underwriter Buddy by SPD,FA")

llm = ChatGroq(api_key=groq_api_key,model="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
If Context not present provide general responses.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions:{input}
"""
)

def vector_embeddings():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3")
        st.session_state.loader=PyPDFDirectoryLoader(".S/us_census")# Data Ingestion
        st.session_state.docs = st.session_state.loader.load()## Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## CHunk Creation
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])## splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)## vector store

# We can give the chatbot history so that it remembers the queries and responses for better interaction but it will slow down the chatbot
    # if "history" not in st.session_state:
    #     st.session_state.history = []


prompt1 = st.text_input("Enter your question from the documents:")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("Vector store DB is ready")




if prompt1:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    if "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response["answer"])

        # With streamlit expander
        with st.expander("Show more"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("----------------------------------")
    else:
        response = llm.invoke(prompt1)
        st.write("Response time:", time.process_time() - start)
        st.write(response.content)
