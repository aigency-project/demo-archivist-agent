import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from pathlib import Path

# Función para cargar los documentos PDF en un directorio
def load_pdf_documents(pdf_dir: str):
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    documents = []
    for path in pdf_paths:
        try:
            loader = PyMuPDFLoader(path)
        except Exception:
            loader = UnstructuredPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
    return documents

# Función para construir el vectorstore
def build_vectorstore(documents, persist_dir: str):
    # Definir cómo dividir los documentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n", "\u200c", " "])
    chunks = splitter.split_documents(documents)
    
    # Crear las incrustaciones (embeddings) usando HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Crear y persistir el vectorstore con Chroma
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    vectordb.persist()  # Guardar el vectorstore en el directorio
    return vectordb

# Función para obtener el vectorstore persistido si existe
def get_autodesk_vectorstore(persist_directory: str) -> Chroma:
    """
    Cargar un Chroma DB persistido si existe, de lo contrario, falla rápidamente.
    """
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db_path = Path(persist_directory)
    if db_path.exists() and any(db_path.iterdir()):
        # Si existe el vectorstore persistido, cargarlo
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedder,
        )
    else:
        raise FileNotFoundError(
            f"No se encontró un vectorstore en {persist_directory!r}. "
            "Por favor, ejecute primero el paso de construcción (por ejemplo, usando build_vectorstore)."
        )

# Ejemplo de uso

# Directorio donde están los PDFs
pdf_directory = "./"
# Directorio donde se guardará el vectorstore
persist_directory = "./persisted_vectorstore"

# Cargar documentos desde el directorio PDF
#documents = load_pdf_documents(pdf_directory)
#print(f"Se cargaron {len(documents)} documentos.")
#
## Crear el vectorstore y guardarlo
#vectordb = build_vectorstore(documents, persist_directory)

# Cargar el vectorstore persistido (si ya existe)
try:
    loaded_vectordb = get_autodesk_vectorstore(persist_directory)
    print("Vectorstore cargado exitosamente.")
except FileNotFoundError as e:
    print(str(e))
