import chromadb
# from chromadb.config import Settings  # Removed unused import
import os
import numpy as np
import uuid
from typing import List, Any
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

class VectorStore:
    def __init__(self, collection_name: str="pptx_docs", persist_directory: str="./data/vector_store"):
        """Initialize the vector store.
        Args:
            collection_name (str): Name of the collection in the vector store.
            persist_directory (str): Directory to persist the vector store.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the ChromaDB client and collection."""
        # create persist directory if not exists
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        #Get or create collection
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description":"pptx docs embeddings for RAG"})
        print(f"Collection {self.collection_name} initialized in {self.persist_directory}")

    
    # def add_docs(self, documents: list[Any], embeddings: np.ndarray):
    #     """Add documents and their embeddings to the vector store.
    #     Args:
    #         documents (list): List of langchain document.
    #         embeddings (list): List of corresponding embeddings.
    #     """
    #     if len(documents) != len(embeddings):
    #         raise ValueError("Documents and embeddings must have the same length.")
        
    #     print(f"Adding {len(documents)} documents to the collection {self.collection_name}.")
        
    #     # Prepare data for ChormaDB
    #     ids = []
    #     metadata = []
    #     documents_text = []
    #     embeddings_list = []
    #     for i, (doc, emb) in enumerate(zip(documents, embeddings)):
    #         doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
    #         ids.append(doc_id)
    #         #Prepare metadata
    #         metadata=dict(doc.metadata)
    #         metadata['doc_index'] = i
    #         metadata['content_length'] = len(doc.page_content)
    #         metadata.append(metadata)
    #         #document content
    #         documents_text.append(doc.page_content)
    #         #embedding
    #         embeddings_list.append(emb.tolist())
            
    #     # Add to collection
    #     try:
    #         self.collection.add(
    #             ids=ids,
    #             documents=documents_text,
    #             embeddings=embeddings_list,
    #             metadatas=metadata
    #         )
    #         print(f"Successfully added {len(documents)} documents to the collection {self.collection_name}.")

    #     except Exception as e:
    #         print(f"Error adding documents to the collection: {e}")
    #         raise


    def create_id(self, doc, index):
        file_path = doc.metadata.get('file_path', 'unknown')
        slide_idx = doc.metadata.get('slide_index', index)
        doc_id = f"{file_path}_slide_{slide_idx}"
        # print(f"File path from metadata: {file_path}")
        # print(f"Slide index from metadata: {slide_idx}")
        # print(f"Generated document ID: {doc_id}")
        return doc_id
    
    def add_docs(self, docs, embeddings):
        """
        Add documents and their embeddings to the vector store.
        Args:
            docs (list): List of document objects with metadata.
            embeddings (list): List of embedding vectors.
        """
        metadatas = []
        contents = []
        ids = []
        for i, doc in enumerate(docs):
            # doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(self.create_id(doc, i))
            #Prepare metadata
             # Flatten metadata: convert non-primitive values to strings
            meta = {}
            for k, v in dict(doc.metadata).items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    meta[k] = v
                else:
                    meta[k] = json.dumps(v)

            meta['doc_index'] = i
            meta['content_length'] = len(doc.page_content)
            metadatas.append(meta)
            # metadatas.append(doc.metadata)
            contents.append(doc.page_content)

            # Check which IDs already exist in the collection
            existing = set(self.collection.get(ids=ids)["ids"])
            new_ids, new_docs, new_embeds, new_metas = [], [], [], []
            for idx, doc_id in enumerate(ids):
                if doc_id in existing:
                    # Optionally, delete the old entry before updating
                    self.collection.delete(ids=[doc_id])
                new_ids.append(doc_id)
            new_docs.append(contents[idx])
            new_embeds.append(embeddings[idx])
            new_metas.append(metadatas[idx])

         # Add only new or updated docs
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )