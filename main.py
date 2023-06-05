import os
import dotenv

import streamlit as st
import pandas as pd

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import (
    ENTITY_MEMORY_CONVERSATION_TEMPLATE,
)
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


dotenv.load_dotenv()

documents = [
    {
        "id": 1,
        "question": "How to reset a password?",
        "answer": "To reset your password, click on the 'Forgot password?' "
                  "link on the login page. Enter your email address, "
                  "and we will send you instructions to reset your password.",
        "url": "https://example.com/confluence/recover-password",
    },
    {
        "id": 2,
        "question": "How to contact support?",
        "answer": "You can contact our support team by sending an email to "
                  "support@example.com or by calling +1 (123) 456-7890.",
        "url": "https://example.com/confluence/contact-support",
    },
    {
        "id": 3,
        "question": "How to set up two-factor authentication?",
        "answer": "To set up two-factor authentication, go to the "
                  "'Security Settings' section of your account and "
                  "follow the instructions.",
        "url": "https://example.com/confluence/2fa-setup",
    },
]

df = pd.DataFrame(documents)
df.head()

loader = DataFrameLoader(df, page_content_column="question")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0
)
texts = text_splitter.split_documents(documents)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


db = FAISS.from_documents(texts, embeddings)
db.as_retriever()
db.save_local("faiss_index")


if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


def get_text():
    """
    Get the user input text.
    Returns:
         (str): The text entered by the user
    """
    input_text = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="Your AI assistant here! Ask me anything ...",
        label_visibility="hidden",
    )
    return input_text


def new_chat():
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        user_message = "User:" + st.session_state["past"][i]
        bot_response = "Bot:" + st.session_state["generated"][i]

        user_doc = Document(page_content=user_message)
        bot_doc = Document(page_content=bot_response)
        db.add_documents([user_doc, bot_doc])

    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state["stored_session"].append(st.session_state["generated"])
    st.session_state.entity_memory.buffer.clear()


st.title("Memory bot")


api = st.sidebar.text_input("API-Key", type="password")
MODEL = st.sidebar.selectbox(
    label="Model",
    options=["gpt-3.5-turbo", "text-davinci-003", "text-davinci-002"],
)

if api:
    llm = OpenAI(
        temperature=0,
        openai_api_key=api,
        model_name=MODEL,
    )

    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(
            llm=llm, k=10
        )

    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory,
    )
else:
    st.error("No API found")
st.sidebar.button("New Chat", on_click=new_chat, type="primary")

user_input = get_text()

if user_input:
    output = Conversation.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="🐢")
        st.success(st.session_state["generated"][i], icon="🤖")
