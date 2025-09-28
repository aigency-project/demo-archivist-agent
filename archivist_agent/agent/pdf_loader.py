"""
PDF document processing and RAG system for the Archivist Agent.

This module provides functionality to load PDF documents, create vector embeddings,
and query a local RAG (Retrieval-Augmented Generation) system.
"""

import os
import glob
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_chroma import Chroma
from transformers import pipeline

# Configure logger
logger = logging.getLogger(__name__)

# Configuration constants
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "/app/code/agent/pdfs")
PERSIST_DIRECTORY = "/app/code/agent/persisted_vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-small"

# Text processing constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_CONTEXT_LENGTH = 1000
MAX_NEW_TOKENS = 150
RETRIEVAL_K = 3

# Global singletons (lazy initialization)
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_hf_pipeline: Optional[Any] = None
_vectorstore: Optional[Chroma] = None

def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Get or create the embedding model (singleton pattern)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info(f"Initialized embedding model: {EMBEDDING_MODEL_NAME}")
    return _embedding_model

def _get_hf_pipeline() -> Any:
    """Get or create the HuggingFace pipeline (singleton pattern)."""
    global _hf_pipeline
    if _hf_pipeline is None:
        _hf_pipeline = pipeline(
            "text2text-generation",
            model=GENERATION_MODEL_NAME,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            truncation=True
        )
        logger.info(f"Initialized generation pipeline: {GENERATION_MODEL_NAME}")
    return _hf_pipeline

def _load_pdf_documents(pdf_dir: str) -> List[Any]:
    """Load PDF documents from a directory."""
    os.makedirs(pdf_dir, exist_ok=True)
    
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    logger.info(f"Searching for PDFs in: {pdf_dir}")
    logger.info(f"Found {len(pdf_paths)} PDF files")
    
    documents = []
    for path in pdf_paths:
        try:
            logger.debug(f"Loading PDF: {path}")
            # Try PyMuPDFLoader first (faster and more reliable)
            loader = PyMuPDFLoader(path)
            docs = loader.load()
        except Exception as e:
            logger.warning(f"PyMuPDFLoader failed for {path}, trying UnstructuredPDFLoader: {e}")
            try:
                loader = UnstructuredPDFLoader(path)
                docs = loader.load()
            except Exception as e2:
                logger.error(f"Failed to load PDF {path}: {e2}")
                continue
        
        documents.extend(docs)
        logger.debug(f"Loaded {len(docs)} pages from {path}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def _build_vectorstore(documents: List[Any], persist_dir: str) -> Chroma:
    """Build and persist the vectorstore from documents."""
    if not documents:
        raise ValueError("No documents provided to build vectorstore")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n", "\u200c", " "]
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    vectorstore = Chroma.from_documents(
        chunks, 
        _get_embedding_model(), 
        persist_directory=persist_dir
    )
    
    logger.info(f"Created vectorstore with {len(chunks)} chunks in {persist_dir}")
    return vectorstore

def _clean_vectorstore_directory(persist_directory: str) -> None:
    """Clean the vectorstore directory completely."""
    try:
        if os.path.exists(persist_directory):
            logger.info(f"Cleaning corrupted directory: {persist_directory}")
            shutil.rmtree(persist_directory)
            logger.info("Directory cleaned successfully")
    except Exception as e:
        logger.error(f"Error cleaning directory: {e}")

def _load_existing_vectorstore(persist_directory: str) -> Optional[Chroma]:
    """Load vectorstore if it exists, otherwise return None."""
    db_path = Path(persist_directory)
    
    if not (db_path.exists() and any(db_path.iterdir())):
        return None
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=_get_embedding_model(),
        )
        logger.info("Loaded existing vectorstore from persistence")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        
        # Clean corrupted database
        if _is_database_error(e):
            logger.info("Database corruption detected, cleaning vectorstore...")
            _clean_vectorstore_directory(persist_directory)
        
        return None

def _is_database_error(error: Exception) -> bool:
    """Check if the error is related to database issues."""
    error_str = str(error).lower()
    db_error_indicators = ["disk i/o error", "database error", "error getting collection"]
    return any(indicator in error_str for indicator in db_error_indicators)

def get_or_create_vectorstore() -> Chroma:
    """Get the global vectorstore, loading or creating it if necessary."""
    global _vectorstore
    
    if _vectorstore is not None:
        return _vectorstore
    
    logger.info("Initializing vectorstore...")
    
    # Try to load existing vectorstore
    _vectorstore = _load_existing_vectorstore(PERSIST_DIRECTORY)
    
    if _vectorstore is None:
        logger.info("Creating vectorstore from PDF documents...")
        documents = _load_pdf_documents(PDF_DIRECTORY)
        
        if not documents:
            error_msg = (
                f"No PDF documents found in {PDF_DIRECTORY}. "
                f"To use the RAG system, place PDF files in: {os.path.abspath(PDF_DIRECTORY)}"
            )
            logger.warning(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Ensure directory is clean before creating
        _clean_vectorstore_directory(PERSIST_DIRECTORY)
        
        try:
            _vectorstore = _build_vectorstore(documents, PERSIST_DIRECTORY)
            logger.info("Vectorstore created successfully")
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            # Clean and retry once
            _clean_vectorstore_directory(PERSIST_DIRECTORY)
            _vectorstore = _build_vectorstore(documents, PERSIST_DIRECTORY)
            logger.info("Vectorstore recreated successfully after error")
    
    return _vectorstore

def reset_vectorstore() -> None:
    """Reset the vectorstore completely and clear global variables."""
    global _vectorstore
    _vectorstore = None
    _clean_vectorstore_directory(PERSIST_DIRECTORY)
    logger.info("Vectorstore reset completely")

def _build_context_from_docs(docs: List[Any], max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """Build context string from retrieved documents."""
    if not docs:
        return ""
    
    context = "\n\n".join([doc.page_content for doc in docs])
    return context[:max_length] if len(context) > max_length else context

def _format_prompt(context: str, query: str) -> str:
    """Format the prompt for the T5 model."""
    if context.strip():
        return f"Answer based on context: {context}... Question: {query}"
    return f"Answer: {query}"

def _generate_response(prompt: str) -> str:
    """Generate response using the HuggingFace pipeline."""
    try:
        hf_pipeline = _get_hf_pipeline()
        response = hf_pipeline(prompt)
        
        if isinstance(response, list) and response:
            return response[0].get('generated_text', str(response))
        return str(response)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Error generating response with the model."

def _retrieve_documents(query: str, max_retries: int = 1) -> Tuple[List[Any], str]:
    """
    Retrieve documents from vectorstore with automatic recovery.
    
    Args:
        query: Search query
        max_retries: Maximum number of retry attempts
    
    Returns:
        Tuple of (documents, error_message). Error message is empty on success.
    """
    for attempt in range(max_retries + 1):
        try:
            vectorstore = get_or_create_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            docs = retriever.invoke(query)
            logger.debug(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            return docs, ""
            
        except Exception as e:
            if attempt == 0 and _is_database_error(e):
                logger.info("Database error detected, attempting recovery...")
                try:
                    reset_vectorstore()
                    continue
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")
            
            error_msg = f"Error retrieving documents: {e}"
            logger.error(error_msg)
            return [], error_msg
    
    return [], "Failed to retrieve documents after recovery attempt"

def query_rag_system_local(query: str) -> Dict[str, Any]:
    """
    Query the local RAG system.
    
    Args:
        query: Question to ask the RAG system.
    
    Returns:
        Dictionary with 'result' (generated answer) and 'source_documents' (retrieved docs).
    """
    if not query.strip():
        return {
            "result": "Please provide a valid query.",
            "source_documents": []
        }
    
    logger.info(f"Processing RAG query: {query[:100]}...")
    
    # Retrieve relevant documents
    docs, error_msg = _retrieve_documents(query)
    
    if error_msg:
        return {
            "result": f"Error processing query: {error_msg}",
            "source_documents": []
        }
    
    if not docs:
        return {
            "result": "No relevant documents found for your query.",
            "source_documents": []
        }
    
    # Build context and generate response
    context = _build_context_from_docs(docs)
    prompt = _format_prompt(context, query)
    result = _generate_response(prompt)
    
    logger.info(f"Generated response with {len(docs)} source documents")
    
    return {
        "result": result.strip() if isinstance(result, str) else str(result),
        "source_documents": docs
    }

# Public API functions
__all__ = [
    "query_rag_system_local",
    "get_or_create_vectorstore", 
    "reset_vectorstore"
]

