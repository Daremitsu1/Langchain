import streamlit as st
import os
import requests
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger')

def loadPDFFromLocal(pdf_file_path='D:\Python Projects\POC\Radium\Ana\documents\Compassionate Leave Revised.pdf'):
    loader = UnstructuredPDFLoader(pdf_file_path)
    loaded_docs = loader.load()
    return loaded_docs

def splitDocument(loaded_docs):
    # Splitting documents into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_docs = splitter.split_documents(loaded_docs)
    return chunked_docs

def createEmbeddings(chunked_docs):
    # Create embeddings and store them in a FAISS vector store
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunked_docs, embedder)
    return vector_store

@st.cache_data()
def load_nltk_data():
    nltk.download('punkt')

load_nltk_data()

def loadLLMModel():
    llm=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0.9, "max_length":512})
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def askQuestions(vector_store, chain, question):
    # Ask a question using the QA chain
    similar_docs = vector_store.similarity_search(question)
    response = chain.run(input_documents=similar_docs, question=question)
    return response

chain = loadLLMModel()

PDF_loaded_docs = loadPDFFromLocal()
PDF_chunked_docs = splitDocument(PDF_loaded_docs)
PDF_vector_store = createEmbeddings(PDF_chunked_docs)

st.title('🦜🔗 + 🦙 Colab Anna Bot..')

question = st.text_input('Ask your question')

# If the user hits enter
if question:
    # Then pass the prompt to the LLM
    PDF_response = askQuestions(PDF_vector_store, chain, question)
    # ...and write it out to the screen
    st.write(PDF_response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = PDF_vector_store.similarity_search_with_score(question) 
        # Write out the first 
        st.write(search[0][0].page_content) 