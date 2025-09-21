# gen-ai-challenge-1
Created a RAG system. It will load documents and generate the result based on query releated to those docuemnts.

## Components
1. Data ingestion pipeline
2. Data retrievel pipeline
3. Vector store
4. Functional Model

## To start application.
```Python
 streamlit run chatbot_ui.py
```
## To import dependencies 
```Pyhton
pip install -r requirements.txt
```

#### Data location:
./data/pptx

#### Vector Store location:
./data/vector_store

#### Ollama used for:
- Embedder: mxbai-embed-large
- FM: llama3.2