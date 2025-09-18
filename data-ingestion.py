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

# Step 1 ::Load pptx files from a directory
directory_path = "./data/pptx"
document = process_all_pptx(directory_path)

## Step 2 ::split the documents into smaller chunks
from chunks import split_documents
chunks = split_documents(document)
chunks

## convert the text to embeddings and store in a vector db
texts = [doc.page_content for doc in chunks]

## intialize the embedder
from embedder import Embedder
embeddder = Embedder()
embeddder
##Generate the Embeddings
embeddings = embeddder.generate_embeddings(texts)

##store into a vector db
from vectorstore import VectorStore
vectorstore = VectorStore()
vectorstore
vectorstore.add_docs(chunks, embeddings)
