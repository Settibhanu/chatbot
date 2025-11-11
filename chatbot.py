import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import faiss # Although imported, this is only needed if you manually interact with faiss structure

# Load environment variables from .env file (for API Key)
load_dotenv()

# --- Configuration ---
# Fix 1: Update the embedding model to the fully qualified name (including 'models/') 
# to resolve the "unexpected model name format" error (400).
EMBEDDING_MODEL = "models/text-embedding-004" 
LLM_MODEL = "gemini-2.5-pro" 

# Streamlit app title
st.title("Smart AI Chatbot with RAG")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.session_state.is_processing = False

def initialize_embeddings_model():
    """Initializes the Google Generative AI Embeddings model."""
    try:
        # Using the recommended, non-deprecated model with correct prefix
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Error initializing embeddings model: {e}")
        raise

def initialize_vectorstore(embedding_model):
    """
    Load the existing FAISS index from local storage.
    NOTE: If the index doesn't exist, this will raise a FileNotFoundError.
    """
    try:
        # Load the local index using the new embedding model
        vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        st.session_state.vectorstore = vectorstore
        st.success("Vector store loaded successfully.")
    except Exception as e:
        # We catch all exceptions here, including FileNotFoundError if the index is missing
        st.warning(f"Vector store not found or could not be loaded. Please upload a PDF to create it. ({e})")

def setup_retriever():
    """Set up the retriever for the vector store."""
    if st.session_state.vectorstore:
        return st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return None

def setup_llm():
    """Initialize the Language Model (LLM) for generating responses."""
    try:
        return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        raise

def setup_prompt_template():
    """Set up the system and human prompt template for the chat."""
    system_prompt = (
        "You are an assistant for question-answering tasks based on a provided document. "
        "Use the following retrieved context to answer the question. "
        "If the context does not contain the answer, say that you don't know based on the provided documents. "
        "Keep the answer concise, limited to two or three sentences."
        "\n\n"
        "Context: {context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

def process_query(query, retriever, llm, prompt):
    """Process the user query through the retrieval and generation pipeline."""
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return response

# --- Sidebar for File Upload and Index Creation ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Placeholder for the part that *creates* the embeddings and index
    if uploaded_file and not st.session_state.is_processing:
        st.session_state.is_processing = True
        try:
            with st.spinner("Processing PDF and creating embeddings... This may take a moment."):
                # 1. Load the PDF
                with open("temp_doc.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader("temp_doc.pdf")
                documents = loader.load()

                # 2. Split documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)

                # 3. Initialize embeddings (using the updated model)
                embeddings = initialize_embeddings_model()
                
                # 4. Create and save the FAISS index (this is where the embed_content calls happen!)
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                st.session_state.vectorstore.save_local("faiss_index")
            
            st.success("PDF successfully processed and FAISS index created!")
            
        except Exception as e:
            st.error(f"Error during file processing: {e}")
        finally:
            st.session_state.is_processing = False
            # Rerun the app to update the state and load the new vector store
            st.rerun()

# --- Main Application Logic ---

# 1. Initialize embeddings model
try:
    embeddings_model = initialize_embeddings_model()
except Exception:
    st.stop()

# 2. Load or initialize vector store
if st.session_state.vectorstore is None:
    # Attempt to load from disk on initial run
    initialize_vectorstore(embeddings_model)

# 3. Set up retriever, LLM, and prompt template
retriever = setup_retriever()
llm = setup_llm()
prompt = setup_prompt_template()

# 4. Chat interface
if st.session_state.vectorstore is None:
    st.info("Please upload a PDF file on the sidebar to begin chatting.")
else:
    # Display existing chat history
    for chat in st.session_state.chat_history:
        if "user" in chat:
            with st.chat_message("user"):
                st.write(chat['user'])
        elif "assistant" in chat:
            with st.chat_message("assistant"):
                st.write(chat['assistant'])
    
    # Process new query
    query = st.chat_input("Ask something about the uploaded document: ")
    
    if query:
        st.session_state.chat_history.append({"user": query})
        with st.chat_message("user"):
            st.write(query)
        
        with st.spinner("Processing your request..."):
            try:
                response = process_query(query, retriever, llm, prompt)
                assistant_response = response["answer"]
                st.session_state.chat_history.append({"assistant": assistant_response})

                with st.chat_message("assistant"):
                    st.write(assistant_response)

            except Exception as e:
                # Catch API errors here. If you still see the quota error, refer to the next steps.
                st.error(f"An error occurred during query processing. Check your API key and quotas. Error: {e}")