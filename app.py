import streamlit as st
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

st.set_page_config(page_title="Customer Support Bot", page_icon="🤖")

LANG = {
    "zh": {
        "page_title": "客服机器人",
        "header": "🤖 智能客服问答机器人",
        "input_label": "请输入你的问题：",
        "input_placeholder": "例如：如何退货？",
        "spinner": "正在检索答案...",
        "response_title": "客服回答：",
    },
    "en": {
        "page_title": "Customer Support Bot",
        "header": "🤖 Smart Customer Support Q&A",
        "input_label": "Please enter your question:",
        "input_placeholder": "e.g., How to return a product?",
        "spinner": "Retrieving answer...",
        "response_title": "Bot Answer:",
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    lang_choice = st.selectbox("🌐 Language / 语言", ("中文", "English"))
    st.session_state.lang = "zh" if lang_choice == "中文" else "en"

lang = st.session_state.lang
text = LANG[lang]

st.title(text["header"])

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_faq():
    df = pd.read_excel("UP_Wiki.xlsx")
    
    documents = []
    for _, row in df.iterrows():
        title = str(row.get('Title', '')).strip()
        content = str(row.get('Content', '')).strip()
        if content:
            documents.append(Document(page_content=content, metadata={"title": title}))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(documents, embeddings)
    return db

retriever = load_faq().as_retriever(search_kwargs={"k": 5})

query = st.text_input(text["input_label"], placeholder=text["input_placeholder"])

if query:
    with st.spinner(text["spinner"]):
        docs = retriever.get_relevant_documents(query)
        if docs:
            st.success(text["response_title"])
            for i, doc in enumerate(docs, start=1):
                title = doc.metadata.get("title", "")
                st.markdown(f"### {i}. {title}")
                st.markdown(doc.page_content)
        else:
            st.warning("没有找到相关内容。" if lang == "zh" else "No relevant content found.")