import streamlit as st
import os
import json
import numpy as np
from dotenv import load_dotenv, find_dotenv
from pathlib import Path 
import torch
import re
from text_embeddings import get_weaviate_client 
import google.generativeai as genai
import weaviate
from weaviate.classes.query import Filter 
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')


@st.cache_resource
def load_models_and_indexes():
    """
    Loads models and indexes required for the chatbot.
    Replaced FAISS text and image indexes with Weaviate for unified retrieval.

    Returns:
        tuple:
            - text_embeddings_model (HuggingFaceEmbeddings): model for text embedding.
            - weaviate_chunks_collection (weaviate.Collection): Weaviate collection for text and image metadata similarity search.
            - articles_data (list): List of article data loaded from JSON (for fallback).
    """
    
    app_dir = Path(__file__).parent
    text_embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    weaviate_client = get_weaviate_client()
    if not weaviate_client:
        st.error("Failed to initialize Weaviate client. Please check your configuration.")
        st.stop()
    
    try:
        weaviate_chunks_collection = weaviate_client.collections.get("ArticleChunk")
    except Exception as e:
        st.error(f"Error getting Weaviate 'ArticleChunk' collection. Ensure schema is created: {e}")
        st.stop() 

    articles_data_path = "data/processed/merged_articles_with_images.json"
    if not os.path.exists(articles_data_path):
        st.error(f"Article data file not found at {articles_data_path}. Please ensure it exists.")
        st.stop()
    with open(articles_data_path, "r", encoding="utf-8") as f:
        articles_data = json.load(f)
    return text_embeddings_model, weaviate_chunks_collection, articles_data


def is_valid_image_url(url):
    """
    Checks if the given URL is valid for loading an image.

    Args:
        url (str): URL to check.

    Returns:
        bool: True if the URL starts with http/https and is not a base64 SVG or GIF image.
    """
    return url and url.startswith(("http://", "https://")) and not url.lower().startswith(("data:image/svg+xml", "data:image/gif;base64"))


def retrieve_content(query, text_embeddings_model, weaviate_chunks_collection, articles_data, top_k_text=2):
    """
    Retrieves the most relevant text documents and their associated image URLs from Weaviate.
    This single function handles both text and image retrieval from the unified Weaviate index.

    Args:
        query (str): User's text query.
        text_embeddings_model (HuggingFaceEmbeddings): model for generating text embeddings for the query.
        weaviate_chunks_collection (weaviate.Collection): Weaviate collection for similarity search.
        articles_data (list): fallback list of articles to return in case of failure.
        top_k_text (int): number of top relevant text documents (unique articles) to return.

    Returns:
        tuple: (list of dict, str or None)
            - List of relevant documents with keys 'content', 'title', 'url', and 'image_urls'.
            - URL of the most relevant image found across all retrieved chunks, or None.
    """
    query_embedding = text_embeddings_model.embed_query(query)
    
    unique_articles_with_images = []
    seen_urls = set()
    top_image_url = None 

    try:
        response = weaviate_chunks_collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k_text * 5,
            return_properties=[
                "text_content", "article_title", "article_url", 
                "original_article_image_urls", "image_descriptions", "chunk_unique_id"
            ]
        )

        for o in response.objects:
            url = o.properties.get("article_url")
            title = o.properties.get("article_title", "No Title")
            content = o.properties.get("text_content", "")
            image_urls_for_chunk = o.properties.get("original_article_image_urls", [])
            if not isinstance(image_urls_for_chunk, list):
                image_urls_for_chunk = [image_urls_for_chunk] 
            if not top_image_url:
                for img_url in image_urls_for_chunk:
                    if is_valid_image_url(img_url):
                        top_image_url = img_url
                        break 
            if url and url not in seen_urls:
                unique_articles_with_images.append({
                    "content": content,
                    "title": title, 
                    "url": url,
                    "image_urls": image_urls_for_chunk 
                })
                seen_urls.add(url)
            if len(unique_articles_with_images) >= top_k_text and top_image_url:
                break 
        
        return unique_articles_with_images, top_image_url

    except Exception as e:
        st.error(f"Error during Weaviate content retrieval: {e}. Falling back to raw articles.")
        fallback_articles = []
        seen_urls_fallback = set()
        for article in articles_data:
            if article["url"] not in seen_urls_fallback:
                fallback_articles.append({
                    "content": article["content"] + "...", 
                    "title": article.get("title", "No Title"), 
                    "url": article["url"],
                    "image_urls": article.get("image_urls", []) 
                })
                seen_urls_fallback.add(article["url"])
            if len(fallback_articles) >= top_k_text:
                break
        return fallback_articles, None 

text_embeddings_model, weaviate_chunks_collection, articles_data = load_models_and_indexes()

if __name__ == "__main__":
    st.title("The Batch Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about DeepLearning.AI The Batch articles..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            chat_history_str = ""
            for msg in st.session_state.messages[-10:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = re.sub(r'!?\[.*?\]\(https?://[^\s)]+\)|https?://[^\s)]+', '', msg["content"])
                content = content.split("For more information, click on the links below:")[0]
                content = content.split("Here is a relevant image:")[0] 
                content = content.replace("\n\n---\n\n", "")
                chat_history_str += f"{role}: {content.strip()}\n"

            relevant_content_docs, top_image_url = retrieve_content(
                prompt, text_embeddings_model, weaviate_chunks_collection, articles_data, top_k_text=2
            )

            context_text = ""
            unique_article_links = {}
            for doc in relevant_content_docs:
                context_text += f"Content: {doc['content']}\nTitle: {doc['title']}\n\n"
                if 'url' in doc:
                    unique_article_links[doc['url']] = doc['title']

            article_links_output = ""
            if unique_article_links:
                for url, title in unique_article_links.items():
                    article_links_output += f"- **{title}** ([Read More]({url}))\n"
# The structured prompt for Gemini Model remains the same as previously agreed
            full_prompt = f"""
### Role and Goal
You are an expert on articles, providing helpful, comprehensive, and accurate answers.
Your primary goal is to provide a detailed and accurate answer to the user's question, *strictly based on the provided article context*. If the context does not contain enough information to fully answer, clearly state that you cannot answer based on the given information.

### Constraints
- Do NOT include any links, URLs, markdown links (e.g., [text](url)), or direct references to article titles within your main answer. These will be handled separately at the end of the response.
- Do not make up information. Only use the provided context.
- Do not answer not related for articles questions.

### Context from Articles
{context_text}

### Relevant Image Information
The user is also being shown one relevant image that was found for their query. You may briefly acknowledge its presence if it's highly relevant to your answer (e.g., "A related image is also displayed below which illustrates X.").

### Conversation History
To maintain context and continuity, your responses should be informed by the current conversation history.
{chat_history_str}

### User Question
{prompt}

### Task
Provide your detailed and accurate answer based on the above information.
"""

        try:
            gemini_response = model.generate_content(full_prompt, generation_config=genai.GenerationConfig(temperature=0.2))
            assistant_reply = re.sub(r'!?\[.*?\]\(https?://[^\s)]+\)|https?://[^\s)]+', '', gemini_response.text)
            assistant_reply = assistant_reply.strip() 
        except Exception as e:
            assistant_reply = f"An error occurred while generating the response: {e}"

        if top_image_url:
            assistant_reply += "\n\n---\n\nHere is a relevant image:\n"
            assistant_reply += f"![Relevant Image]({top_image_url})\n\n"

        if article_links_output:
            assistant_reply += "\n\n---\n\nFor more information, click on the links below:\n" + article_links_output
        elif not relevant_content_docs and not top_image_url: 
            assistant_reply += "\n\n---\n\nI couldn't find specific articles or relevant images for your query in my database. Please try rephrasing your question."
        elif not relevant_content_docs and top_image_url: 
            assistant_reply += "\n\n---\n\nI couldn't find specific articles to answer your question, but here is a related image."

        st.markdown(assistant_reply)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})