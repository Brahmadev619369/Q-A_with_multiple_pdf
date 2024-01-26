import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
import os
def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunk(row_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunk = text_splitter.split_text(row_text)
    return chunk


def get_vectorstore(text_chunk):
    embeddings = OpenAIEmbeddings(openai_api_key = os.getenv("OPENAI_API_KEY"))
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector = FAISS.from_texts(text_chunk,embeddings)
    return vector


def get_conversation_chain(vectorstores):
    llm = ChatOpenAI(openai_api_key = os.getenv("OPENAI_API_KEY"))
    # llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key = "chat_history",return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vectorstores.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def user_input(user_question):
    response = st.session_state.conversation({"question":user_question})
    st.session_state.chat_history = response["chat_history"]

    for indx, msg in enumerate(st.session_state.chat_history):
        if indx % 2==0:
            st.write(user_template.replace("{{MSG}}",msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)



def main():
    # load secret key
    load_dotenv()
    
    # config the pg
    st.set_page_config(page_title="Chat with multiple PDFs" ,page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your docs")
    if user_question:
        user_input(user_question)

    # st.write(user_template.replace("{{MSG}}","Hello Robot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}","Hello Human"), unsafe_allow_html=True)

    # create side bar
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_doc = st.file_uploader(label="Upload your documents",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner(text="Processing"):

            # get pdf text
                row_text = get_pdf_text(pdf_doc)
            # get the text chunk
                text_chunk = get_text_chunk(row_text)
                # st.write(text_chunk)
            # create vecor store
                vectorstores = get_vectorstore(text_chunk)
                # st.write(vectorstores)
            # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstores)


if __name__ == "__main__":
    main()