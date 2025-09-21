"""
Herramientas RAG simples para Google ADK.

Dos herramientas A2A: añadir documentos y consultarlos.
"""

__version__ = "1.0.0"

from .tools import add_document_to_knowledge_base, query_knowledge_base

__all__ = [
    'add_document_to_knowledge_base',
    'query_knowledge_base'
]