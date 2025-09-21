#!/usr/bin/env python3
"""
Script para ejecutar el ejemplo desde cualquier directorio.
"""

import os
import sys
import tempfile

# Añadir el directorio actual al path para importar tools
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tools import add_document_to_knowledge_base, query_knowledge_base

def main():
    print("=== Herramientas RAG Simples para Google ADK ===\n")
    
    # Crear un documento de prueba simple
    test_content = """# Google ADK

Google ADK es una plataforma de desarrollo que permite crear aplicaciones con IA.

## Características
- Autenticación OAuth 2.0
- APIs de procesamiento de lenguaje natural
- Herramientas de desarrollo integradas
- Soporte para múltiples lenguajes de programación

## Configuración
Para configurar Google ADK, necesitas:
1. Crear una cuenta de desarrollador
2. Obtener las credenciales de API
3. Instalar el SDK
4. Configurar la autenticación
"""
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    
    try:
        # Ejemplo 1: Añadir un documento
        print("1. Añadiendo documento de prueba...")
        result = add_document_to_knowledge_base(test_file)
        
        if result["success"]:
            print(f"✓ Documento añadido: {result['chunks_added']} chunks en {result['processing_time']}s")
        else:
            print(f"✗ Error: {result['message']}")
            return
        
        print()
        
        # Ejemplo 2: Consultar la base de conocimiento
        print("2. Consultando base de conocimiento...")
        result = query_knowledge_base("¿Cómo configurar autenticación en Google ADK?", top_k=2)
        
        if result["success"] and result["results"]:
            print(f"✓ Encontrados {result['results_count']} resultados:")
            for i, doc in enumerate(result["results"], 1):
                print(f"\n   Resultado {i} (score: {doc['score']:.3f}):")
                print(f"   Fuente: {doc['source']}")
                print(f"   Contenido: {doc['content'][:150]}...")
        else:
            print(f"✗ No se encontraron resultados: {result['message']}")
        
        print("\n✓ Ejemplo completado exitosamente!")
        
    finally:
        # Limpiar archivo temporal
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == "__main__":
    main()