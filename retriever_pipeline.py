from typing import List, Any, Dict
from vectorstore import VectorStore
from embedder import Embedder

class RAGRetrieverPipeline:
    def __init__(self, vectorstore: VectorStore, embedder: Embedder):
        """Initialize the RAG Retriever Pipeline with a vector store and an embedder."""
        self.vectorstore = vectorstore
        self.embedder = embedder

    def answer_query(self, query:str, top_k:int=5, score_threshold:float=0.0) -> List[Dict[str, Any]]:
        """
        Answer a query using the RAG approach.
        Args:
            query (str): The input query string.
            top_k (int): Number of top documents to retrieve.
            score_threshold (float): Minimum score threshold for retrieved documents.
            Returns:
            str: The generated answer.
        """
        print(f"Answering query: {query}")
        print(f"Retrieving top {top_k} documents with score threshold {score_threshold}")

        #Generate embedding for the query
        query_embedding = self.embedder.generate_embeddings([query])[0]

        #Search in the vector store
        results = self.vectorstore.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )

        #Process results
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            ids = results['ids'][0]

            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                # convert distance to sililarity score (ChromaDB uses cosine distance)
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': i + 1
                    })

            print(f"Retrieved doc {i+1}: ID={doc_id}, Score={similarity_score:.4f}")
        else:
            print("No documents retrieved.")

        return retrieved_docs
    
# /rag_retriever_pipeline = RAGRetrieverPipeline(vectorstore=VectorStore(), embedder=Embedder())
# rag_retriever_pipeline
