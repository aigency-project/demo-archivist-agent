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
    test_content = """# 

Caso de Archivo #1
ID de Caso: 78B Título del Caso: El Caso del "Lienzo Silencioso" Fecha: 12 de Octubre de
2008 Estado: Sin Resolver (Caso Frío)
Resumen del Crimen: Se perpetró un robo en la Galería de Arte Moderno de la ciudad. La
obra sustraída fue "Noche Estrellada sobre el Río"
, del aclamado pintor contemporáneo
Javier Soto, valorada en 3 millones de euros. El robo ocurrió durante la noche, sin que
sonara ninguna de las alarmas de presión, movimiento o láser.
Modus Operandi (M.O.): El ladrón demostró un conocimiento técnico extremadamente
avanzado. Las alarmas fueron neutralizadas mediante un dispositivo que emitía una
frecuencia sónica específica, descalibrando los sensores temporalmente sin dejar rastro en
el sistema. El cristal blindado de la vitrina fue cortado con una herramienta de diamante de
alta precisión, dejando un círculo perfecto. El autor del robo no utilizó la fuerza bruta en
ningún momento.
Personas de Interés:
●
●
Sospechoso Principal: Un individuo no identificado conocido por el alias "El
Maestro"
. Se cree que es un ladrón de guante blanco con experiencia en sistemas
de seguridad de alta gama. No se obtuvieron huellas dactilares ni ADN.
Testigos Clave: Ninguno. El guardia de seguridad de turno no reportó ninguna
anomalía.
●
Víctima: La Galería de Arte Moderno y el coleccionista privado dueño de la obra.
Evidencia Clave:
●
●
●
Fragmentos microscópicos de un cristal de cuarzo cerca de la vitrina, posiblemente
de un emisor sónico personalizado.
El corte circular en el cristal blindado, que sugiere el uso de una herramienta
especializada no comercial.
Registros de seguridad corruptos en un bucle de 30 minutos, justo durante el tiempo
estimado del robo.
Análisis Forense: El análisis del cristal de cuarzo no arrojó ninguna coincidencia en las
bases de datos comerciales. El laboratorio concluyó que el dispositivo emisor fue
probablemente de fabricación casera y de un diseño único. El análisis del sistema de
seguridad reveló que la corrupción de los registros se hizo de forma remota, explotando una
vulnerabilidad de día cero en el firmware del sistema, desconocida hasta la fecha.
Resolución: El caso sigue sin resolver. La obra de arte nunca fue recuperada y la identidad
de "El Maestro" permanece como un misterio. El caso es un ejemplo paradigmático de
robos de alta tecnología.
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
        result = query_knowledge_base("Dame informacion sobre robos")
        
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