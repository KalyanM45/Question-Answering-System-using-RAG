import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
groq_api_key = os.getenv('API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Document Question Answering System", layout="wide")
st.title("Document Question Answering System")
st.caption("Initially Ingest the Data into Vector Store and then ask questions.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader("./Artifacts")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:20])
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Ingest the Data into Vector Store"):
    vector_embedding()
    st.write("Data is Ingested in vector store database. You can now ask questions.")

if prompt1:
    try:
        if "vectors" in st.session_state:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write(f"Response time: {time.process_time() - start} seconds")
            st.write(response['answer'])
    except:
        st.write("Please Ingest Data First. Click on the button - Ingest the Data into Vector Store")


