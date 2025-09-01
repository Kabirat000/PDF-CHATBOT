import os
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------------------------------------
# Streamlit App UI
# -------------------------------------------------
st.set_page_config(page_title="📚 Chat with your PDF", layout="wide")
st.title("📚 Chat with your PDF using Groq + LangChain")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded PDF temporarily
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    # Create a unique Chroma directory for each upload
    persist_dir = f"./chroma_store/{uuid.uuid4().hex}"

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    db = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=persist_dir
    )
    retriever = db.as_retriever(search_kwargs={"k": 8})

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key
    )

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"   # 👈 FIX applied here
    )

    # Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"   # 👈 FIX applied here too
    )

    # -------------------------------------------------
    # Multi-turn Chat UI
    # -------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display past messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("❓ Ask a question about your PDF")
    if query:
        # Show user message
        st.chat_message("user").markdown(query)
        st.session_state["messages"].append({"role": "user", "content": query})

        # Run the chain
        result = qa_chain.invoke({"question": query})
        answer = result["answer"]

        # Show assistant message
        with st.chat_message("assistant"):
            st.markdown(f"**🧠 Answer:**\n\n{answer}")

            # Sources
            pages = {doc.metadata.get("page", "Unknown page") for doc in result["source_documents"]}
            st.markdown("**📄 Sources:** " + ", ".join([f"Page {p}" for p in sorted(pages)]))

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )
