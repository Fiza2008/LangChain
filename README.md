# 🚀 RAG Chatbot using LangChain + OpenAI + PDF Loader

An end-to-end Retrieval Augmented Generation (RAG) chatbot that allows users to upload PDFs and ask questions based on the content.

## 🔥 Features
- Document ingestion (PDFs)
- Text chunking + embeddings using OpenAI
- Vector search using LangChain FAISS (default)
- Prompt engineering with custom system prompts
- Chat interface using Streamlit

## 🏗️ Architecture


## 🛠 Tech Stack
| Component | Tool |
|----------|------|
| LLM | OpenAI GPT Model |
| Framework | LangChain |
| Embeddings | OpenAI Embeddings |
| UI | Streamlit |
| Storage | Local Vector DB (FAISS) |

## 🚧 Setup Instructions

```bash
git clone https://github.com/<your-username>/rag-langchain-openai
cd rag-langchain-openai
pip install -r requirements.txt

create .env 

OPENAI_API_KEY="your_key"

Run APP

streamlit run app.py

✅ Output

A chat UI where user uploads PDF → system answers from content.

📌 Future Enhancements
Replace FAISS with Pinecone
Add PDF summarization


---

### ✅ Code Boilerplate

📁 Repository Structure:

rag-langchain-openai/
│── app.py
│── requirements.txt
│── utils/
│ └── rag_pipeline.py
│── .env
│── README.md



**app.py**

```python
import streamlit as st
from utils.rag_pipeline import get_rag_response

st.title("📚 RAG Chatbot using LangChain + OpenAI")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
query = st.text_input("Ask a question")

if uploaded and query:
    answer = get_rag_response(uploaded, query)
    st.write(answer)


----

**utils/rag_pipeline.py**

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def get_rag_response(uploaded_pdf, query):
    loader = PyPDFLoader(uploaded_pdf)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4.1-mini"),
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    return qa.run(query)

