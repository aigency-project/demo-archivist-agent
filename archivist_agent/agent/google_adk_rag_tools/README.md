# Herramientas RAG Simples para Google ADK

Dos herramientas A2A simples para trabajar con documentación de Google ADK:
- Añadir documentos a una base de conocimiento local
- Consultar la base de conocimiento

Todo funciona offline con modelos ligeros.

## Instalación

```bash
pip install sentence-transformers chromadb PyMuPDF
```

## Prueba Rápida

```bash
cd google_adk_rag_tools/
python run_example.py
```

## Uso

```python
# Importar desde el directorio local
from tools import add_document_to_knowledge_base, query_knowledge_base

# Añadir documento
result = add_document_to_knowledge_base("/ruta/al/documento.pdf")
print(result)

# Consultar
result = query_knowledge_base("¿Cómo configurar autenticación?")
print(result)
```

## Formatos Soportados

- PDF (.pdf)
- Texto (.txt) 
- Markdown (.md)

## Características

- **Offline**: Sin APIs externas ni claves
- **Ligero**: Modelo de 80MB
- **Rápido**: 1-3 segundos por documento
- **Simple**: Solo 2 herramientas A2A

## Estructura Simplificada

```
google_adk_rag_tools/
├── __init__.py          # Interfaz principal
├── tools.py             # Las 2 herramientas A2A
├── requirements.txt     # Dependencias mínimas
├── run_example.py       # Ejemplo ejecutable
└── README.md           # Este archivo
```

## Ejecución

### Opción 1: Automática (Recomendada)
```bash
cd google_adk_rag_tools/
python run_example_auto.py
```

### Opción 2: Manual
```bash
cd google_adk_rag_tools/
pip install -r requirements.txt
python run_example.py
```

### Si hay problemas con sentence-transformers:
```bash
python fix_dependencies.py
```

O usa la versión simple:
```bash
pip install scikit-learn PyMuPDF
python -c "from tools_simple import *"
```

## Solución de Problemas

- **Error con sentence-transformers**: Ejecuta `python fix_dependencies.py`
- **Versión simple**: Usa `tools_simple.py` con TF-IDF en lugar de embeddings
- **Verificar dependencias**: Ejecuta `python check_dependencies.py`

Eso es todo. Simple y funcional.