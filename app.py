import os
import uuid
import hashlib
import streamlit as st
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# Qdrant-specific imports
from langchain_community.vectorstores import Qdrant
from qdrant_client.http.models import Distance, VectorParams

# Aggressive performance optimizations
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
# Prioritize Streamlit secrets over environment variables
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------------------------------------
# Streamlit App UI
# -------------------------------------------------
st.set_page_config(
    page_title="🤖 PDF Chat AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🤖 PDF Chat AI")
st.caption("Chat with your documents using advanced AI")

# Simple sidebar
with st.sidebar:
    st.header("🎯 Controls")

    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state["messages"] = []
        st.rerun()

    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        st.metric("💬 Messages", len(st.session_state["messages"]))

    st.divider()
    st.subheader("⚙️ Settings")
    search_k = st.slider("Document chunks", 2, 5, 3)
    temperature = st.slider("AI creativity", 0.0, 0.3, 0.05, 0.05)

# Simple file uploader
st.subheader("📄 Upload Your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Generate hash for document caching
    file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()
    cache_key = f"vectorstore_{file_hash}"

    # Store the current cache key in session state
    st.session_state["current_cache_key"] = cache_key

    # Check if this document is already processed
    if cache_key in st.session_state:
        st.success("✅ Document ready! Using cached version.")
        qa_chain = st.session_state[cache_key]
    else:
        with st.spinner("🚀 Processing your PDF document..."):
            # Save uploaded PDF temporarily
            pdf_path = f"uploaded_{file_hash}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load PDF
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()

            # Ultra-fast text splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " "],
                keep_separator=False
            )
            documents = splitter.split_documents(docs)

            # Ultra-fast embedding model configuration
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    # 'normalize_embeddings': True,
                    'trust_remote_code': False
                },
                encode_kwargs={
                    'batch_size': 64,
                    # 'show_progress_bar': False,
                    'convert_to_numpy': True,
                    'normalize_embeddings': True
                },
                cache_folder="./model_cache"
            )

            # Initialize Qdrant
            persist_dir = f"./qdrant_store/{file_hash}"
            os.makedirs(persist_dir, exist_ok=True)

            vectors_config = VectorParams(size=384, distance=Distance.COSINE)
            collection_name = "pdf_documents"

            db = Qdrant.from_documents(
                documents,
                embedding_model,
                path=persist_dir,
                collection_name=collection_name,
                vectors_config=vectors_config
            )

            retriever = db.as_retriever(
                search_kwargs={
                    "k": min(search_k if 'search_k' in locals() else 3, 3),
                    # "fetch_k": 6,
                    "score_threshold": 0.3
                }
            )

            # Ultra-fast LLM configuration
            llm = ChatGroq(
                model="moonshotai/kimi-k2-instruct",
                api_key=SecretStr(groq_api_key) if groq_api_key else None,
                temperature=temperature if 'temperature' in locals() else 0.05,
                max_tokens=1000,
                timeout=15,
                max_retries=1,
                streaming=False
            )

            # Conversational Retrieval Chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            # Cache the processed chain
            st.session_state[cache_key] = qa_chain
            st.success("✨ Document processed successfully!")

            # Clean up temporary file
            try:
                os.remove(pdf_path)
            except:
                pass

    # Display document info
    with st.expander("📄 Document Information", expanded=False):
        st.write(f"**File Name:** {uploaded_file.name}")
        st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
        st.write(f"**Status:** ✅ Ready for questions")

# -------------------------------------------------
# ChatGPT-style Chat UI (always available)
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display conversation history like ChatGPT
st.subheader("💬 Conversation")

# Display all messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            st.caption(f"📄 Sources: {message['sources']}")

# Chat input - always available
if prompt := st.chat_input("Ask me anything about your document..."):
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display user message immediately (like ChatGPT)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Minimal chat history for speed
                chat_history = []
                if len(st.session_state["messages"]) >= 3:
                    prev_user = st.session_state["messages"][-3]
                    prev_assistant = st.session_state["messages"][-2]
                    if prev_user["role"] == "user" and prev_assistant["role"] == "assistant":
                        chat_history = [(prev_user["content"], prev_assistant["content"])]

                # Get response using the cached qa_chain from session state
                current_cache_key = st.session_state.get("current_cache_key")
                current_qa_chain = st.session_state.get(current_cache_key) if current_cache_key else None

                if current_qa_chain:
                    result = current_qa_chain.invoke({
                        "question": prompt[:500],
                        "chat_history": chat_history
                    })
                    answer = result["answer"]

                    # Process sources
                    source_pages = [doc.metadata.get("page", "") for doc in result.get("source_documents", [])][:3]
                    sources_text = ", ".join([f"Page {p}" for p in source_pages if p and p != "Unknown"])

                    # Display response
                    st.markdown(answer)
                    if sources_text:
                        st.caption(f"📄 Sources: {sources_text}")

                    # Add assistant message to chat history
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_text
                    })
                else:
                    error_msg = "Error: No document loaded. Please upload a PDF first."
                    st.error(error_msg)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": error_msg
                    })

            except Exception as e:
                error_msg = f"Error: {str(e)[:100]}..."
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg
                })

if uploaded_file is None:
    st.info("👋 Upload a PDF document to start chatting!")
