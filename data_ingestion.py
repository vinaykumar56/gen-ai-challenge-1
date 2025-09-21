from langchain_docling import DoclingLoader
from langchain_community.document_loaders import DirectoryLoader
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Function to process all pptx files in a directory
def process_all_pptx(directory_path: str):
    """Process all pptx files in the given directory and its sub-directories.
    Args:
        directory_path (str): Path to the directory containing pptx files.
    """
    all_documents = []
    ppt_dir = Path(directory_path)
    
    ppt_files = list(ppt_dir.rglob("*.pptx"))
    print(f"Found {len(ppt_files)} pptx files in {directory_path}")
    for ppt_file in ppt_files:
        print(f"Processing file: {ppt_file}")
        loader = DoclingLoader(str(ppt_file))
        documents = loader.load()
        for doc in documents:
            doc.metadata['source'] = str(ppt_file)
            doc.metadata['file_name'] = ppt_file.name
            doc.metadata['file_path'] = str(ppt_file)
            doc.metadata['num_slides'] = len(documents)
            doc.metadata['file_type'] = 'pptx'

        all_documents.extend(documents)
        print(f"Loaded {len(documents)} documents from {ppt_file}")

    print(f"\n Total documents loaded: {len(all_documents)}")
    return all_documents

#create main method to run the ingestion
def main():    
    import os

    # Ensure the data directories exist
    if not os.path.exists("./data"):
        os.makedirs("./data", exist_ok=True)
    if not os.path.exists("./data/vector_store"):
        os.makedirs("./data/vector_store", exist_ok=True)
    if not os.path.exists("./data/pptx"):
        os.makedirs("./data/pptx", exist_ok=True)

    print("Starting data ingestion process...")
    # Step 1 ::Load pptx files from a directory
    directory_path = "./data/pptx"
    document = process_all_pptx(directory_path)
    print(f"Total documents loaded from pptx files: {len(document)}")

    ## Step 2 ::split the documents into smaller chunks
    # from chunks import split_documents
    # chunks = split_documents(document)
    chunks = document  # --- IGNORE --- use full document without splitting into
    # chunks

     ## Step 3 ::intialize the embedder
    from embedder import Embedder
    # ollama embedding model
    embeddder = Embedder(model_name="mxbai-embed-large")
    
    # Initialize the vectorstore
    from vectorstore import VectorStore
    vectorstore = VectorStore()

    ## convert the text to embeddings and store in a vector db
    texts = [doc.page_content for doc in chunks]
    print(f"Total texts to be embedded: {len(texts)}")
    
    ##Generate the Embeddings
    embeddings = embeddder.generate_embeddings(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    ##Step 4 :: store into a vector db
    vectorstore.add_docs(chunks, embeddings)
    print("Total documents in the vectorstore:"+str(vectorstore.collection.count()))

    return vectorstore, embeddder


if __name__ == "__main__":
    main()

# main()