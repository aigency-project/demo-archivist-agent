#RAG LOADER
import os
import re
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List
import logging

# Cargar variables de entorno desde .env
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ---------------------------

def load_pdf(file_path: str) -> str:
    """ 
    Reads the text content from a PDF file using PyMuPDF and returns it as a single string.

    Parameters:
    - file_path (str): The file path to the PDF file.

    Returns:
    - str: The concatenated text content of all pages in the PDF.
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_text_by_case(text: str) -> list[str]:
    """
    Splits a text string into a list of substrings, with each substring representing a full case.
    It assumes that each case starts with a line like 'ID de Caso: ...'.

    Parameters:
    - text (str): The input text containing multiple cases.

    Returns:
    - list[str]: A list where each element is the text of a complete case.
    """
    # Split the text by the case identifier. The pattern uses a positive lookahead
    # to keep the delimiter ('ID de Caso:') as part of the resulting chunks.
    chunks = re.split(r'(?=ID de Caso:)', text)
    
    # The first element might be empty if the text starts with the delimiter, so we filter it out.
    # We also strip whitespace from each case text.
    return [chunk.strip() for chunk in chunks if chunk.strip()] 


def split_text(text: str) -> list[str]:
    """
    Splits a text string into a list of non-empty substrings based on paragraphs.

    Parameters:
    - text (str): The input text to be split.

    Returns:
    - list[str]: A list containing non-empty substrings.
    """
    # Split by one or more newlines, which is a more robust way to capture paragraphs
    chunks = re.split(r'\n+', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using the Gemini AI API for document retrieval.
    """
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/text-embedding-004"
        title = "Custom query"
        return genai.embed_content(
            model=model,
            content=input,
            task_type="retrieval_document",
            title=title
        )["embedding"]

def create_chroma_db(documents: List[str], path: str, name: str, file_path: str, file_name: str):
    """
    Creates or loads a Chroma database using the provided documents, path, and collection name.

    Parameters:
    - documents (List[str]): A list of document chunks to be added to the database.
    - path (str): The directory where the Chroma database will be stored.
    - name (str): The name of the collection within the Chroma database.
    - file_path (str): The path to the original file.
    - file_name (str): The name of the original file.

    Returns:
    - Tuple[chromadb.Collection, str]: A tuple containing the Chroma Collection and its name.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=GeminiEmbeddingFunction()
    )

    #model = _get_embedding_model()
    #embeddings = model.encode(chunks)

    # Generate IDs for each document chunk
    ids = [str(i) for i in range(len(documents))]
    metadatas = [{
            "filename": file_name,
            "file_path": file_path,
            "chunk_index": i
        } for i in range(len(documents))]

    # Add documents to the collection in batches
    batch_size = 100  # ChromaDB recommends batch sizes of 100-200 for efficiency
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        logging.info(f"Adding batch {i//batch_size + 1} with {len(batch_docs)} documents...")
        db.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metadatas)
    
    logging.info(f"Successfully created and populated ChromaDB collection '{name}' at '{path}'.")
    return db, name

def load_chroma_collection(path: str, name: str) -> chromadb.Collection:
    """
    Loads an existing Chroma collection from the specified path and name.

    Parameters:
    - path (str): The path where the Chroma database is stored.
    - name (str): The name of the collection.

    Returns:
    - chromadb.Collection: The loaded Chroma Collection.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(
        name=name,
        embedding_function=GeminiEmbeddingFunction()
    )
    return db

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Configuration is now loaded from the .env file.
    # Example .env file:
    # PDF_FILE_PATH="/path/to/your/document.pdf"
    # CHROMA_DB_PATH="./chroma_db"
    # COLLECTION_NAME="my_rag_collection"

    PDF_FILE_PATH = os.getenv("PDF_FILE_PATH")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_rag_collection")
    # ---------------------

    if not PDF_FILE_PATH or not os.path.exists(PDF_FILE_PATH):
        logging.error(f"Error: The file '{PDF_FILE_PATH}' was not found or the path is not set in .env.")
        logging.error("Please create a .env file and set the 'PDF_FILE_PATH' variable.")
    else:
        logging.info(f"1. Loading PDF: {PDF_FILE_PATH}")
        pdf_text = load_pdf(file_path=PDF_FILE_PATH)

        logging.info("2. Splitting text into chunks by case...")
        text_chunks = split_text_by_case(text=pdf_text)
        logging.info(f"   - Found {len(text_chunks)} chunks (cases).")

        logging.info(f"3. Creating and populating ChromaDB collection '{COLLECTION_NAME}'...")
        file_name = os.path.basename(PDF_FILE_PATH)
        create_chroma_db(
            documents=text_chunks,
            path=CHROMA_DB_PATH,
            name=COLLECTION_NAME,
            file_path=PDF_FILE_PATH,
            file_name=file_name
        )

        logging.info("\n--- Ingestion Complete ---")
        logging.info(f"Your document has been processed and stored in the ChromaDB collection.")
        logging.info(f"   - Database Path: {CHROMA_DB_PATH}")
        logging.info(f"   - Collection Name: {COLLECTION_NAME}")