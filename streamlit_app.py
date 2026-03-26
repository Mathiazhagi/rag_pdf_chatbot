import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

st.title("PDF Chatbot (RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("PDF processed successfully!")

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the PDF:")

    if not question:
        st.info("Please ask a question.")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful assistant.

Answer ONLY from the given context.
Do not make up answers.
If the answer is not available, say "Not found in document".

Context:
{context}

Question:
{question}

Answer:
"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm.invoke(prompt)
                answer = response.content
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.subheader("Source Chunks:")
        for i, doc in enumerate(docs, 1):
            st.write(f"Chunk {i}:")
            st.write(doc.page_content[:300] + "...")