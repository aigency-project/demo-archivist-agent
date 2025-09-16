import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# Cargar el vectorstore persistido
PERSIST_DIRECTORY = "./persisted_vectorstore"

def get_autodesk_vectorstore(persist_directory: str) -> ChromaVectorStore:
    """
    Cargar un Chroma DB persistido si existe, de lo contrario, falla rápidamente.
    """
    db_path = Path(persist_directory)
    if not db_path.exists() or not any(db_path.iterdir()):
        raise FileNotFoundError(
            f"No se encontró un vectorstore en {persist_directory!r}. "
            "Por favor, ejecute primero el paso de construcción (por ejemplo, usando build_vectorstore)."
        )

    # Use the ChromaDB client to connect to the persisted database
    db = chromadb.PersistentClient(path=persist_directory)

    # Get the collection
    collection_name = "llama_collection"  # Or the name of your collection
    collection = db.get_or_create_collection(name=collection_name)

    # Create the ChromaVectorStore from the client and collection
    vector_store = ChromaVectorStore(chroma_collection=collection)

    return vector_store

def retrieve_docs(query: str):
    """
    Recupera documentos relevantes basados en una consulta.
    """
    try:
        loaded_vectordb = get_autodesk_vectorstore(PERSIST_DIRECTORY)
        print("Vectorstore cargado exitosamente.")

        storage_context = StorageContext.from_defaults(vector_store=loaded_vectordb)
        index = VectorStoreIndex.from_vector_store(
            vector_store=loaded_vectordb, storage_context=storage_context
        )

        retriever = index.as_retriever(similarity_top_k=6)
        retrieved_docs = retriever.retrieve(query)
        return "\n".join([doc.text for doc in retrieved_docs])

    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

query = "¿Hay casos en fabricas?"
retrieved_docs = retrieve_docs(query)
print(retrieved_docs)