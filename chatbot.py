import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import faiss

# Load environment variables from .env file
load_dotenv()

# Streamlit app title
st.title("Smart AI Chatbot")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.chat_history = []

def initialize_vectorstore():
    """Initialize the vector store and load it into session state if not already done."""
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        st.session_state.vectorstore = vectorstore
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        raise

def setup_retriever():
    """Set up the retriever for the vector store."""
    try:
        return st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    except Exception as e:
        st.error(f"Error setting up retriever: {e}")
        raise

def setup_llm():
    """Initialize the Language Model (LLM) for generating responses."""
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        raise

def setup_prompt_template():
    """Set up the system and human prompt template for the chat."""
    try:
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
    except Exception as e:
        st.error(f"Error setting up prompt template: {e}")
        raise

def process_query(query, retriever, llm, prompt):
    """Process the user query through the retrieval and generation pipeline."""
    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})
        return response
    except Exception as e:
        st.error(f"Error processing query: {e}")
        raise

# Load or initialize vector store
if st.session_state.vectorstore is None:
    try:
        initialize_vectorstore()
    except Exception as e:
        st.error(f"Initialization failed: {e}")

# Set up retriever, LLM, and prompt template
try:
    retriever = setup_retriever()
    llm = setup_llm()
    prompt = setup_prompt_template()
except Exception as e:
    st.error(f"Setup failed: {e}")

# Chat interface
query = st.chat_input("Ask something: ")
if query:
    st.session_state.chat_history.append({"user": query})

    with st.spinner("Processing your request..."):
        try:
            response = process_query(query, retriever, llm, prompt)
            st.session_state.chat_history.append({"assistant": response["answer"]})

            # Display chat history
            for chat in st.session_state.chat_history:
                if "user" in chat:
                    st.write(f"**You:** {chat['user']}")
                elif "assistant" in chat:
                    st.write(f"**Assistant:** {chat['assistant']}")

        except Exception as e:
            st.error(f"An error occurred during query processing: {e}")
