import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from load_rag import load_chroma_collection, GeminiEmbeddingFunction

# Cargar variables de entorno
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ---------------------------

def get_relevant_passages(query: str, db, n_results: int = 5) -> list[str]:
    """
    Retrieves the most relevant passages from the ChromaDB for a given query.

    Parameters:
    - query (str): The user's query.
    - db (chromadb.Collection): The ChromaDB collection object.
    - n_results (int): The number of relevant passages to retrieve.

    Returns:
    - list[str]: A list of the most relevant passages.
    """
    results = db.query(query_texts=[query], n_results=n_results,
            include=['documents', 'metadatas', 'distances', 'embeddings'])
    logging.info(f"--> [TOOL] Results: {results}")
    formatted_results = []
    if results['documents'] and results['documents'][0]:
        logging.info(f"--> [TOOL] Documents: {results['documents'][0]}")
        for i, doc in enumerate(results['documents'][0]):
            logging.info(f"--> [TOOL] Document: {doc}")
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            distance = results['distances'][0][i] if results['distances'] else 1.0
            score = 1.0 - distance  # Convertir distancia a similitud
            
            formatted_results.append({
                "content": doc,
                "score": round(score, 4),
                "source": metadata.get('filename', 'unknown'),
                "file_path": metadata.get('file_path', 'unknown'),
                "chunk_index": metadata.get('chunk_index', 0)
            })
            
    
    logging.info(f"--> [TOOL] Relevant passages retrieved: {formatted_results}")
    return {
            "success": True,
            "query": query.strip(),
            "results_count": len(formatted_results),
            "results": formatted_results,
            #"processing_time": round(time.time() - start_time, 2),
            "message": f"Encontrados {len(formatted_results)} documento(s) relevante(s)" if formatted_results else "No se encontraron documentos relevantes"
        }
    

def make_rag_prompt(query: str, relevant_passages: list[str]) -> str:
    """
    Creates a prompt for the RAG model, combining the query and relevant passages.

    Parameters:
    - query (str): The user's query.
    - relevant_passages (list[str]): A list of relevant passages from the database.

    Returns:
    - str: The formatted prompt.
    """
    # Escape special characters for the prompt
    escaped_passages = [passage.replace("'", "").replace('"', "").replace("\n", " ") for passage in relevant_passages]
    
    # Join the passages into a single string
    context = "\n\n".join(escaped_passages)

    prompt = (
        f"""You are a helpful and informative bot that answers questions using text from the reference passages included below. 
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and conversational tone. 
If the passages are irrelevant to the answer, you may ignore them.

QUESTION: '{query}'

PASSAGES:
{context}

ANSWER: """
    )
    return prompt

def generate_answer(prompt: str) -> str:
    """
    Generates an answer using the Gemini Pro model.

    Parameters:
    - prompt (str): The prompt containing the query and context.

    Returns:
    - str: The generated answer.
    """
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable")
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    answer = model.generate_content(prompt)
    return answer.text

if __name__ == '__main__':
    # --- CONFIGURATION ---
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_rag_collection")
    # ---------------------

    # 1. Load the existing ChromaDB collection
    logging.info(f"Loading ChromaDB collection from {CHROMA_DB_PATH}...")
    db = load_chroma_collection(path=CHROMA_DB_PATH, name=COLLECTION_NAME)
    logging.info("Collection loaded successfully.")

    # 2. Get a query from the user
    user_query = input("\nEnter your question about the documents: ")

    # 3. Retrieve relevant passages
    logging.info(f"\nRetrieving relevant passages for: '{user_query}'...")
    passages = get_relevant_passages(user_query, db)
    logging.info("Passages retrieved.")
    logging.info(passages)

    # 4. Create the prompt
    #rag_prompt = make_rag_prompt(user_query, passages)
    # logging.debug("\n--- RAG PROMPT ---\n", rag_prompt) # Uncomment for debugging

    # 5. Generate the answer
    logging.info("\nGenerating answer...")
    #final_answer = generate_answer(rag_prompt)

    logging.info("\n--- ANSWER ---")
    #logging.info(final_answer)
    logging.info("--------------")