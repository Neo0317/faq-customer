import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from docx import Document
import pandas as pd
from langchain_community.document_loaders import Docx2txtLoader
from PIL import Image
import io
import os

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
        full_text = f"{title}\n\n{content}"  
        documents.append(Document(page_content=full_text, metadata={"source": "UP_Wiki.xlsx"}))

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    split_docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)

    return db

# @st.cache_resource
# def extract_images_from_docx(docx_path):
#     try:
#         doc = Document(docx_path)
#         images = []
#         for rel in doc.part._rels:
#             rel = doc.part._rels[rel]
#             if "image" in rel.target_ref:
#                 img_data = rel.target_part.blob
#                 image = Image.open(io.BytesIO(img_data))
#                 images.append(image)
#         return images
#     except Exception as e:
#         return []

db = load_faq()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=db.as_retriever())

query = st.text_input(text["input_label"], placeholder=text["input_placeholder"])

if query:
    with st.spinner(text["spinner"]):
        answer = qa_chain.run(query)
        st.success(text["response_title"])
        st.write(answer)
