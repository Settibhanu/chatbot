# pdf_loader.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

def load_pdfs_to_vectorstore(pdf_paths, faiss_index_path):
    """
    Load multiple PDFs and create/update the FAISS vector store.

    :param pdf_paths: List of paths to the PDF files.
    :param faiss_index_path: Path to save/load the FAISS index.
    :return: The updated FAISS vector store.
    """
    # Load environment variables from .env file
    load_dotenv()
    try:
        # Create embeddings
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

        for pdf_path in pdf_paths:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            data = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            docs = text_splitter.split_documents(data)

            # Add new documents to the vector store
            vectorstore.add_documents(docs)

        # Save the updated vector store
        vectorstore.save_local(faiss_index_path)

        return vectorstore
    except Exception as e:
        raise Exception(f"Error loading PDFs into vector store: {e}")

if __name__ == "__main__":
    # Define the folder containing PDF files
    pdfs_folder = "pdfs"  # Path to your PDFs folder
    faiss_index_path = "faiss_index"  # Path to your FAISS index file

    # Collect all PDF file paths from the specified folder
    pdf_files = [os.path.join(pdfs_folder, filename) for filename in os.listdir(pdfs_folder) if filename.endswith('.pdf')]

    try:
        updated_vectorstore = load_pdfs_to_vectorstore(pdf_files, faiss_index_path)
        print("PDFs loaded successfully into the vector store.")
    except Exception as e:
        print(f"An error occurred: {e}")
