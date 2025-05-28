import json
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from pathlib import Path


data_path = Path("data/raw/article_data.json")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

FAISS_INDEX_PATH = "data/embeddings/text_embeddings"

def load_articles():
    """
    Load articles from a JSON file and convert them into LangChain Document objects.

    Returns:
        list[Document]: List of documents with article content and metadata.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    return [
        Document(
            page_content=article.get("content", ""), 
            metadata={
                "title": article.get("title", "No Title"),
                "url": article.get("url"),
                "image_url": article.get("image_url") 
            }
        )
        for article in articles
    ]

def create_chunks_from_documents(documents):
    """
    Split each document into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents (list[Document]): The list of full documents.

    Returns:
        list[Document]: A list of smaller chunked documents with preserved metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    chunked_documents = []

    for doc in documents:
        if doc.page_content:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "url": doc.metadata.get("url"),
                            "title": doc.metadata.get("title"),
                            "date": doc.metadata.get("date"),
                            "image_url": doc.metadata.get("image_url")
                        }
                    )
                )
    return chunked_documents

def create_and_save_faiss_index(documents, embeddings_model, index_path):
    """
    Create a FAISS vector index from the list of documents and save it locally.

    Args:
        documents (list[Document]): Chunked documents to index.
        embeddings_model: The embedding model used to encode documents.
        index_path (str): The local directory path to save the FAISS index.
    """
    Path(index_path).mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(documents, embeddings_model)
    db.save_local(index_path)


raw_documents = load_articles()
chunked_docs = create_chunks_from_documents(raw_documents)
create_and_save_faiss_index(chunked_docs, embeddings, FAISS_INDEX_PATH)
