import json
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from weaviate.classes.config import Property, DataType, Configure 
from weaviate.classes.query import Filter 

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY") 

datapath = "/Users/yanakravets/MultRAG/data/processed/merged_articles_with_images.json" 

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

def load_articles() -> list[Document]:
    """Load articles from JSON and convert to LangChain Document objects."""
    try:
        with open(datapath, "r", encoding="utf-8") as f:
            articles = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading articles from {datapath}: {e}")
        return []

    documents = []
    for article in articles:
        doc_content = article.get("content", "")
        img_descriptions = article.get("image_descriptions", [])
        
        documents.append(
            Document(
                page_content=doc_content, 
                metadata={
                    "title": article.get("title", "No Title"),
                    "url": article.get("url", "No URL"),
                    "original_article_image_urls": article.get("image_urls", []), 
                    "image_descriptions": img_descriptions
                }
            )
        )
    return documents


def create_chunks_from_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks with preserved metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_documents = []
    
    for doc_idx, doc in enumerate(documents):
        base_metadata = doc.metadata.copy() 
        
        if doc.page_content:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_unique_id = f"{base_metadata.get('url', 'no_url')}_{doc_idx}_{chunk_idx}"
                chunked_documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **base_metadata,
                            "chunk_unique_id": chunk_unique_id,
                            "chunk_text_length": len(chunk_text),
                        }
                    )
                )
        elif base_metadata.get("image_descriptions"): 
            chunk_unique_id = f"{base_metadata.get('url', 'no_url')}_{doc_idx}_image_only"
            chunked_documents.append(
                Document(
                    page_content="", 
                    metadata={
                        **base_metadata,
                        "chunk_unique_id": chunk_unique_id,
                        "chunk_text_length": 0,
                        "type": "image_only_chunk"
                    }
                )
            )
    return chunked_documents


def get_weaviate_client() -> weaviate.WeaviateClient | None:
    """Establishes a connection to the Weaviate instance using v4 client API."""
    print(f"DEBUG get_weaviate_client: WEAVIATE_URL = {WEAVIATE_URL}")
    print(f"DEBUG get_weaviate_client: WEAVIATE_API_KEY is set: {bool(WEAVIATE_API_KEY)}")
    
    try:
        if WEAVIATE_API_KEY and WEAVIATE_URL and "https://" in WEAVIATE_URL:
            print("Attempting to connect to Weaviate Cloud (WCS) using provided credentials...")
            client = weaviate.connect_to_weaviate_cloud( 
                cluster_url=WEAVIATE_URL,
                auth_credentials=weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY),
                skip_init_checks=True 
            )
        elif WEAVIATE_URL == "http://localhost:8080":
            print("Attempting to connect to local Weaviate instance...")
            client = weaviate.connect_to_local()
        else:
            print("Error: Invalid Weaviate configuration. Neither WCS credentials (HTTPS URL + API Key) nor local (http://localhost:8080) configuration is met.")
            print(f"  Current WEAVIATE_URL: '{WEAVIATE_URL}'")
            print(f"  WEAVIATE_API_KEY is {'set' if WEAVIATE_API_KEY else 'NOT set'}.")
            return None

        client.is_ready() 
        print(f"Successfully connected to Weaviate.")
        return client
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        return None

def define_weaviate_schema(client: weaviate.WeaviateClient, class_name: str = "ArticleChunk"):
    """
    Defines the schema for the ArticleChunk collection (class) in Weaviate using v4 client API.
    Sets 'vectorizer': 'none' as embeddings are pre-computed.
    """
    try:
       
        if client.collections.exists(class_name):
            print(f"Collection '{class_name}' already exists. Skipping creation.")
            return

        client.collections.create(
            name=class_name,
            description="A chunk of an article, including text and image descriptions for multimodal RAG.",
            vectorizer_config=Configure.Vectorizer.none(), 
            properties=[
                Property(name="text_content", data_type=DataType.TEXT, description="The textual content of the chunk."),
                Property(name="article_title", data_type=DataType.TEXT, description="Title of the original article."),
                Property(name="article_url", data_type=DataType.TEXT, description="URL of the original article."),
                Property(name="chunk_unique_id", data_type=DataType.TEXT, description="Unique identifier for this specific chunk."),
                Property(name="original_article_image_urls", data_type=DataType.TEXT_ARRAY, description="URLs of all images in the original article."),
                Property(name="image_descriptions", data_type=DataType.TEXT_ARRAY, description="Descriptions of all images associated with the original article."),
                Property(name="chunk_text_length", data_type=DataType.INT, description="Length of the text content in this chunk."),
            ]
        
        )
        print(f"Collection '{class_name}' created successfully.")

    except Exception as e:
        print(f"Error creating/checking schema for '{class_name}': {e}")


def index_and_save_to_weaviate(class_name: str = "ArticleChunk"):
    """
    Loads articles, creates chunks, generates embeddings, and saves them to Weaviate using v4 client API.
    """
    print("Loading articles...")
    documents = load_articles()
    if not documents: 
        print("No articles loaded. Exiting.")
        return

    print(f"Creating {len(documents)} chunks from articles...")
    chunked_documents = create_chunks_from_documents(documents)
    if not chunked_documents: 
        print("No chunks created. Exiting.")
        return
        
    client = get_weaviate_client()
    if not client: 
        print("Failed to get Weaviate client. Exiting.")
        return

    define_weaviate_schema(client, class_name)
    chunks_collection = client.collections.get(class_name)

    with chunks_collection.batch.dynamic() as batch: 
        for i, doc_chunk in enumerate(chunked_documents):
            text_for_embedding = doc_chunk.page_content
            if doc_chunk.metadata.get("image_descriptions"):
                text_for_embedding += " " + " ".join(doc_chunk.metadata["image_descriptions"])

            try:
                vector = embeddings.embed_query(text_for_embedding)
            except Exception as e:
                print(f"Embedding error for chunk {doc_chunk.metadata.get('chunk_unique_id', i)}: {e}. Skipping.")
                continue

            properties = {
                "text_content": doc_chunk.page_content,
                "article_title": doc_chunk.metadata.get("title", ""),
                "article_url": doc_chunk.metadata.get("url", ""),
                "chunk_unique_id": doc_chunk.metadata.get("chunk_unique_id", f"chunk_{i}"),
                "original_article_image_urls": doc_chunk.metadata.get("original_article_image_urls", []),
                "image_descriptions": doc_chunk.metadata.get("image_descriptions", []),
                "chunk_text_length": doc_chunk.metadata.get("chunk_text_length", 0),
            }
            
            try:
                batch.add_object(
                    properties=properties,
                    vector=vector 
                )
            except Exception as e:
                print(f"Weaviate import error for properties {properties.get('chunk_unique_id', 'N/A')}: {e}. Skipping.")
                
            if (i + 1) % 100 == 0:
                print(f"{i + 1} ")

    print("Batch import complete. Verifying total objects...")
    try:
        chunks_collection = client.collections.get(class_name)
        count_result = chunks_collection.aggregate.over_all(total_count=True)
        total_objects = count_result.total_count
        print(f"Total objects in Weaviate for '{class_name}': {total_objects}")
    except Exception as e:
        print(f"Could not retrieve object count: {e}")

    client.close()
  

if __name__ == "__main__":
    index_and_save_to_weaviate(class_name="ArticleChunk")
