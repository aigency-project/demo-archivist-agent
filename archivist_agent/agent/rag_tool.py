import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import resolve_embed_model  # Importa esta función
from pathlib import Path
from llama_index.core.settings import Settings # Importa el objeto Settings

# Cargar el vectorstore persistido
PERSIST_DIRECTORY = "./persisted_vectorstore"

# Define el modelo de embeddings
# 'sentence-transformers/all-MiniLM-L6-v2' es un modelo open-source muy popular
embed_model = resolve_embed_model("local:sentence-transformers/all-MiniLM-L6-v2")

# Configura el modelo de embeddings globalmente para LlamaIndex
Settings.embed_model = embed_model
# Si también usas un LLM, puedes configurarlo aquí:
# Settings.llm = ...

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

    db = chromadb.PersistentClient(path=persist_directory)
    collection_name = "llama_collection"
    collection = db.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    return vector_store

def retrieve_docs(query: str):
    """
    Recupera documentos relevantes basados en una consulta.
    """
    try:
        loaded_vectordb = get_autodesk_vectorstore(PERSIST_DIRECTORY)
        print("Vectorstore cargado exitosamente.")

        # LlamaIndex ya utiliza el modelo de embeddings configurado globalmente
        # No necesitas pasarlo de nuevo aquí, a menos que quieras anularlo.
        index = VectorStoreIndex.from_vector_store(
            vector_store=loaded_vectordb
        )
        print("Index creado.")


        retriever = index.as_retriever(similarity_top_k=6)
        print("Retriever")
        retrieved_docs = retriever.retrieve(query)
        print("retrieved_docs")
        print(retrieved_docs)
        # return "\n".join([doc.text for doc in retrieved_docs]) # Esto no funciona siempre
        # Mejor usar el método 'get_content'
        return "\n\n---\n\n".join([doc.get_content() for doc in retrieved_docs])


    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

query = "¿Hay casos en fabricas?"
retrieved_docs = retrieve_docs(query)
print(retrieved_docs)