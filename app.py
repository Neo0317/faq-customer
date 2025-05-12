import streamlit as st
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="UP Wiki Assistant", page_icon="📚")

st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .stTextInput > label {
            font-weight: bold;
            font-size: 1.1rem;
        }
        .card {
            padding: 1rem;
            border-radius: 12px;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

LANG = {
    "zh": {
        "page_title": "UP智能助手",
        "header": "📚 UP智能助手",
        "input_label": "🔍 请输入你的问题：",
        "input_placeholder": "例如：如何在Dejavoo刷卡机上删除credit App？",
        "spinner": "📖 正在查找相关内容...",
        "response_title": "📘 检索结果如下：",
    },
    "en": {
        "page_title": "UP Wiki Assistant",
        "header": "📚 UP Wiki Assistant",
        "input_label": "🔍 Please enter your query:",
        "input_placeholder": "e.g., How to delete the credit App on Dejavoo terminals",
        "spinner": "📖 Searching...",
        "response_title": "📘 Results:",
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    lang_choice = st.selectbox("🌐 Language / 语言", ("中文", "English"))
    st.session_state.lang = "zh" if lang_choice == "中文" else "en"
    simplify = st.checkbox("简洁模式 / Concise mode", value=True)

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

@st.cache_resource
def get_qa_chain():
    db = load_faq()
    retriever = db.as_retriever(search_kwargs={"k": 3})  # 取最相关的3条
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

retriever = load_faq().as_retriever(search_kwargs={"k": 5})
qa_chain = get_qa_chain()

query = st.text_input(text["input_label"], placeholder=text["input_placeholder"])

if query:
    with st.spinner(text["spinner"]):
        if simplify:
            answer = qa_chain.run(query)
            st.success(text["response_title"])
            st.markdown(answer)
        else:
            docs = retriever.get_relevant_documents(query)
            if docs:
                st.success(text["response_title"])
                for i, doc in enumerate(docs, start=1):
                    title = doc.metadata.get("title", "")
                    st.markdown(f"### {i}. {title}")
                    st.markdown(doc.page_content)
            else:
                st.warning("没有找到相关内容。" if lang == "zh" else "No relevant content found.")