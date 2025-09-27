import os
import logging
from retrieval import (
    load_chroma_collection,
    get_relevant_passages,
    make_rag_prompt,
    generate_answer,
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ---------------------------

def search_case_archives(query: str) -> str:
    """
    Searches the Central Knowledge Base (CKB) for information related to a query.

    This tool connects to a ChromaDB vector store, retrieves relevant passages
    from historical case files, and uses a generative model to synthesize an
    answer based on the retrieved context.

    Args:
        query: The natural language query to search for in the archives.

    Returns:
        A string containing the synthesized answer from the generative model.
    """
    # --- CONFIGURATION ---
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_rag_collection")
    # ---------------------

    logging.info(f"--> [TOOL] Loading ChromaDB collection: {COLLECTION_NAME}")
    db = load_chroma_collection(path=CHROMA_DB_PATH, name=COLLECTION_NAME)

    logging.info(f"--> [TOOL] Retrieving relevant passages for query: '{query}'")
    passages = get_relevant_passages(query, db)
    logging.info(f"--> [TOOL] Relevant passages retrieved: {passages}")
    

    #logging.info("--> [TOOL] Creating RAG prompt.")
    #rag_prompt = make_rag_prompt(query, passages)

    #logging.info("--> [TOOL] Generating answer.")
    #final_answer = generate_answer(rag_prompt)

    return passages