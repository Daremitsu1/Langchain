# Import dependencies
import streamlit as st
# Environment Keys
# from dotenv import load_dotenv
# Import PDF dependencies
from PyPDF2 import PdfReader
# Import Langchain dependencies
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmltemplate import css, bot_template, user_template
import os

# OpenAI service key
os.environ['OPENAI_API_KEY'] = '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'

# HuggingFace service key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

with st.sidebar:
    st.image('https://colab.nexaei.com/assets/images/logo.png')
    st.info('This application allows you to use LLMs from OpenAI/HuggingFace for question-answering.')
    selected_model = st.sidebar.radio('Choose your model', ['Hugging Face', 'OpenAI'])
    st.write(selected_model)
    # Model information
    if selected_model == 'Hugging Face':
        st.markdown('''
        Hugging Face provides a diverse collection of language models that can be used for various natural language processing tasks.
        The models in Hugging Face are free to use and are of much precise question answering based.
        May take a while to fetch answers.
        ''')
    elif selected_model == 'OpenAI':
        st.markdown('''
        OpenAI is an artificial intelligence research laboratory that focuses on developing advanced natural language processing models.
        The models are generally commercial in nature and will provide much elongated answers.
        Answers are fetched pretty quick.
        ''')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        return None

    embeddings = OpenAIEmbeddings()
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except:
        return None

def get_conversation_chain(vectorstore, selected_model):
    if selected_model == 'OpenAI':
        llm = ChatOpenAI()
    else:
        llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, selected_model):
    if st.session_state.conversation_chain is None:
        st.error("Conversation chain is not initialized. Please process the PDFs first.")
        return

    response = st.session_state.conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.title('🤖 Colab Anna Bot..')
    st.write(css, unsafe_allow_html=True)

    # Initialize session state attributes
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    pdf_docs = st.file_uploader('Upload your pdfs here and click on Process 🚀!', accept_multiple_files=True)

    if st.button('Process'):
        with st.spinner('Processing'):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation_chain = get_conversation_chain(vectorstore, selected_model)

    user_question = st.text_input('Enter your question prompt here: 👨🏻‍💻')
    if user_question:
        handle_userinput(user_question, selected_model)

if __name__ == '__main__':
    main()