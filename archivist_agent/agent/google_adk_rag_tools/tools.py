"""
Herramientas RAG simples para documentación de Google ADK.

Dos herramientas A2A: añadir documentos y consultarlos.
Todo funciona offline con modelos ligeros.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Decorador para herramientas A2A
def tool(func):
    """Decorador simple para herramientas A2A."""
    func._is_tool = True
    return func

# Imports simples
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

# Variables globales simples
_embedding_model = None
_chroma_client = None
_collection = None

def _get_embedding_model():
    """Carga el modelo de embeddings de forma lazy."""
    global _embedding_model
    if _embedding_model is None:
        if SentenceTransformer is None:
            raise Exception("sentence-transformers no está instalado. Instala con: pip install sentence-transformers")
        
        try:
            print("Cargando modelo de embeddings (puede tardar unos segundos la primera vez)...")
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Modelo cargado correctamente")
        except Exception as e:
            raise Exception(f"Error cargando modelo de embeddings: {str(e)}")
    return _embedding_model

def _get_chroma_collection():
    """Obtiene o crea la colección de ChromaDB."""
    global _chroma_client, _collection
    
    if _collection is None:
        if chromadb is None:
            raise Exception("chromadb no está instalado. Instala con: pip install chromadb")
        
        # Crear directorio si no existe
        persist_dir = "./vectorstore_data"
        os.makedirs(persist_dir, exist_ok=True)
        
        # Cliente persistente
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Obtener o crear colección
        try:
            _collection = _chroma_client.get_collection("google_adk_docs")
        except:
            _collection = _chroma_client.create_collection("google_adk_docs")
    
    return _collection

def _extract_text(file_path: str) -> str:
    """Extrae texto de un archivo."""
    if not os.path.exists(file_path):
        raise Exception(f"Archivo no encontrado: {file_path}")
    
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        if fitz is None:
            raise Exception("PyMuPDF no está instalado. Instala con: pip install PyMuPDF")
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    
    elif ext in ['.txt', '.md', '.markdown']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    else:
        raise Exception(f"Formato no soportado: {ext}. Soportados: .pdf, .txt, .md")

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Divide texto en chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Buscar punto de corte natural
            for i in range(end - 1, start + chunk_size - 100, -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end >= len(text):
            break
        
        start = max(end - overlap, start + 1)
    
    return chunks

@tool
def add_document_to_knowledge_base(file_path: str) -> dict:
    """
    Añade un documento a la base de conocimiento.
    
    Args:
        file_path: Ruta al archivo (PDF, TXT, MD)
    
    Returns:
        dict: Resultado de la operación
    """
    start_time = time.time()
    
    try:
        # Validación básica
        if not file_path or not isinstance(file_path, str):
            return {
                "success": False,
                "message": "file_path debe ser una cadena válida",
                "chunks_added": 0,
                "processing_time": 0.0
            }
        
        # Extraer texto
        text = _extract_text(file_path)
        if not text:
            return {
                "success": False,
                "message": "No se pudo extraer texto del archivo",
                "chunks_added": 0,
                "processing_time": round(time.time() - start_time, 2)
            }
        
        # Crear chunks
        chunks = _chunk_text(text)
        
        # Generar embeddings
        model = _get_embedding_model()
        embeddings = model.encode(chunks)
        
        # Obtener colección
        collection = _get_chroma_collection()
        
        # Eliminar documento existente si existe
        file_name = os.path.basename(file_path)
        try:
            existing = collection.get(where={"filename": file_name})
            if existing['ids']:
                collection.delete(ids=existing['ids'])
        except:
            pass
        
        # Añadir chunks
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        metadatas = [{
            "filename": file_name,
            "file_path": file_path,
            "chunk_index": i
        } for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "success": True,
            "message": f"Documento '{file_name}' procesado exitosamente",
            "chunks_added": len(chunks),
            "processing_time": round(time.time() - start_time, 2),
            "file_name": file_name,
            "file_path": file_path
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error procesando documento: {str(e)}",
            "chunks_added": 0,
            "processing_time": round(time.time() - start_time, 2)
        }

@tool
def query_knowledge_base(query: str, top_k: int = 3) -> dict:
    """
    Consulta la base de conocimiento.
    
    Args:
        query: Consulta de texto
        top_k: Número de resultados a retornar
    
    Returns:
        dict: Resultados de la búsqueda
    """
    start_time = time.time()
    
    try:
        # Validación básica
        if not query or not isinstance(query, str) or not query.strip():
            return {
                "success": False,
                "message": "query debe ser una cadena no vacía",
                "results_count": 0,
                "results": [],
                "processing_time": 0.0
            }
        
        if top_k < 1 or top_k > 20:
            top_k = 3
        
        # Generar embedding de la consulta
        model = _get_embedding_model()
        query_embedding = model.encode([query.strip()])
        
        # Buscar en la colección
        collection = _get_chroma_collection()
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, collection.count()),
            include=['documents', 'metadatas', 'distances']
        )
        
        # Formatear resultados
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
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
        
        return {
            "success": True,
            "query": query.strip(),
            "results_count": len(formatted_results),
            "results": formatted_results,
            "processing_time": round(time.time() - start_time, 2),
            "message": f"Encontrados {len(formatted_results)} documento(s) relevante(s)" if formatted_results else "No se encontraron documentos relevantes"
        }
        
    except Exception as e:
        return {
            "success": False,
            "query": query.strip() if query else "",
            "results_count": 0,
            "results": [],
            "processing_time": round(time.time() - start_time, 2),
            "message": f"Error en consulta: {str(e)}"
        }