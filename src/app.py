import streamlit as st
import os
import json
import numpy as np
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import torch
import re

import google.generativeai as genai
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import CLIPModel, CLIPProcessor


load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')


def load_models_and_indexes():
    """
    Loads models and indexes required for the chatbot.

    Returns:
        tuple:
            - text_embeddings_model (HuggingFaceEmbeddings): model for text embedding.
            - text_vectorstore (FAISS): FAISS index for text similarity search.
            - clip_model (CLIPModel): CLIP model for image and text embeddings.
            - clip_processor (CLIPProcessor): Processor for the CLIP model.
            - media_index (faiss.Index): FAISS index for image embeddings.
            - media_urls (list): List of image URLs corresponding to media_index.
            - articles_data (list): List of article data loaded from JSON.
    """
    text_embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    text_index_path = "data/embeddings/text_embeddings"
    text_vectorstore = None

    if os.path.exists(text_index_path) and Path(text_index_path).is_dir():
        try:
            text_vectorstore = FAISS.load_local(text_index_path, text_embeddings_model, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Failed to load Langchain FAISS index: {e}")
            st.stop()
    else:
        st.error(f"Text FAISS index directory not found at {text_index_path}.")
        st.stop()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    media_index_path = "data/embeddings/media_embeddings.faiss"
    media_urls_path = "data/embeddings/media_urls.json"

    if not os.path.exists(media_index_path) or not os.path.exists(media_urls_path):
        st.error("Media FAISS index or URLs file not found.")
        st.stop()

    media_index = faiss.read_index(media_index_path)
    with open(media_urls_path, "r", encoding="utf-8") as f:
        media_urls = json.load(f)

    with open("data/raw/article_data.json", "r", encoding="utf-8") as f:
        articles_data = json.load(f)

    return text_embeddings_model, text_vectorstore, clip_model, clip_processor, media_index, media_urls, articles_data


def is_valid_image_url(url):
    """
    Checks if the given URL is valid for loading an image.

    Args:
        url (str): URL to check.

    Returns:
        bool: True if the URL starts with http/https and is not a base64 SVG or GIF image.
    """
    return url and url.startswith(("http://", "https://")) and not url.lower().startswith(("data:image/svg+xml", "data:image/gif;base64"))


def retrieve_text(query, text_embeddings_model, text_vectorstore, articles_data, top_k=3):
    """
    Retrieves the most relevant text documents from the FAISS index based on the user query.

    Args:
        query (str): User's text query.
        text_embeddings_model (HuggingFaceEmbeddings): model for generating text embeddings.
        text_vectorstore (FAISS): FAISS index for similarity search.
        articles_data (list): fallback list of articles to return in case of failure.
        top_k (int): number of top relevant documents to return.

    Returns:
        list of dict: List of relevant documents with keys 'content', 'title', and 'url'.
    """
    query_embedding = text_embeddings_model.embed_query(query)
    try:
        docs = text_vectorstore.similarity_search_by_vector(np.array(query_embedding).astype('float32'), k=top_k)
        return [{"content": doc.page_content, "title": doc.metadata.get("title", "No Title"), "url": doc.metadata.get("url")} for doc in docs]
    except Exception as e:
        st.warning(f"Error during text retrieval, falling back to sample articles: {e}")
        return [{"content": article["content"] + "...", "title": article["title"], "url": article["url"]} for article in articles_data[:top_k]] 


def retrieve_images(query, clip_model, clip_processor, media_index, media_urls):
    """
    Retrieves the most relevant image URL for a given text query using CLIP embeddings.

    Args:
        query (str): User's text query.
        clip_model (CLIPModel): CLIP model.
        clip_processor (CLIPProcessor): CLIP processor.
        media_index (faiss.Index): FAISS index for image embeddings.
        media_urls (list): List of image URLs corresponding to the index.

    Returns:
        str or None: URL of the most relevant image, or None if not found.
    """
    inputs = clip_processor(text=query, return_tensors="pt")
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    query_embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    D, I = media_index.search(query_embedding.cpu().numpy().astype('float32'), 1)  

    idx = I[0][0]
    if 0 <= idx < len(media_urls) and is_valid_image_url(media_urls[idx]):
        return media_urls[idx]
    return None


text_embeddings_model, text_vectorstore, clip_model, clip_processor, media_index, media_urls, articles_data = load_models_and_indexes()

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
            content = content.split("Here are some relevant images:")[0]
            content = content.replace("\n\n---\n\n", "")
            chat_history_str += f"{role}: {content.strip()}\n"

        relevant_text_docs = retrieve_text(prompt, text_embeddings_model, text_vectorstore, articles_data, top_k=2)

        context_text = ""
        unique_article_links = {} 
        for doc in relevant_text_docs:
            context_text += f"Content: {doc['content']}\nTitle: {doc['title']}\n\n"
            if 'url' in doc:
                unique_article_links[doc['url']] = doc['title']

        article_links_output = ""
        if unique_article_links:
            for url, title in unique_article_links.items():
                article_links_output += f"- **{title}** ([Read More]({url}))\n"


        top_image_url = retrieve_images(prompt, clip_model, clip_processor, media_index, media_urls)

        full_prompt = f"""
You are an expert on DeepLearning.AI The Batch articles, providing helpful and comprehensive answers.
Your goal is to provide a detailed and accurate answer to the user's question, *strictly based on the provided article context*. If the context does not contain enough information to fully answer, clearly state that.

IMPORTANT: Do NOT include any links, URLs, markdown links (e.g., [text](url)), or direct references to article titles within your main answer. These will be handled separately at the end of the response.

The user is also being shown one relevant image that was found for their query. You can briefly acknowledge its presence if relevant (e.g., "A related image is also displayed below.").

Your responses should be informed by the current conversation history to maintain context and continuity.

Chat History:
{chat_history_str}

Context from articles:
{context_text}

User question: {prompt}
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
        elif not relevant_text_docs and not top_image_url:
            assistant_reply += "\n\n---\n\nI couldn't find specific articles or relevant images for your query in my database. Please try rephrasing your question."
        elif not relevant_text_docs and top_image_url: 
            assistant_reply += "\n\n---\n\nI couldn't find specific articles to answer your question, but here is a related image."

        st.markdown(assistant_reply)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
