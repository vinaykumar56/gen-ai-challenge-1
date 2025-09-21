
## Text splitting code
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into smaller chunks.
    Args:
        documents (List[Document]): List of langchain Document objects.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    Returns:
        List[Document]: List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, separators=["\n\n", "\n", " ", ""])
    split_docs = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content)
        for i, split in enumerate(splits):
            metadata = dict(doc.metadata)
            metadata['chunk_index'] = i
            metadata['source'] = doc.metadata.get('source', 'unknown')
            split_docs.append(Document(page_content=split, metadata=metadata))
    return split_docs