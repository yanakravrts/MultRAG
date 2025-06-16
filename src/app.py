import streamlit as st
import os
import json
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import re
from text_embeddings import get_weaviate_client
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity  

load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')


@st.cache_resource
def load_models_and_indexes():
    text_embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    weaviate_client = get_weaviate_client()
    if not weaviate_client:
        st.stop()

    weaviate_chunks_collection = weaviate_client.collections.get("ArticleChunk")

    articles_data_path = "data/processed/merged_articles_with_images.json"
    if not os.path.exists(articles_data_path):
        st.stop()

    with open(articles_data_path, "r", encoding="utf-8") as f:
        articles_data = json.load(f)

    return text_embeddings_model, weaviate_chunks_collection, articles_data


def is_valid_image_url(url):
    return url and url.startswith(("http://", "https://")) and not url.lower().startswith(("data:image/svg+xml", "data:image/gif;base64"))


def retrieve_content(query, text_embeddings_model, weaviate_chunks_collection, top_k_text=7):
    """
    Retrieves the most relevant articles and finds the most semantically relevant image.
    If none of the text chunks are sufficiently relevant to the query, no image is returned.
    """
    query_embedding = text_embeddings_model.embed_query(query)

    unique_articles_with_images = []
    seen_urls = set()
    top_image_url = None
    top_image_similarity = -1.0

    response = weaviate_chunks_collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k_text,
        return_properties=[
            "text_content", "article_title", "article_url",
            "original_article_image_urls", "image_descriptions", "chunk_unique_id"
        ]
    )

    if not response.objects:
        return [], None

    context_matches = False
    for o in response.objects:
        content = o.properties.get("text_content", "")
        if content.strip(): 
            content_embedding = text_embeddings_model.embed_query(content)
            similarity = cosine_similarity([query_embedding], [content_embedding])[0][0]
            if similarity > 0.5:  
                context_matches = True
                break

    if not context_matches:
        return [], None

    for o in response.objects:
        url = o.properties.get("article_url")
        title = o.properties.get("article_title", "No Title")
        content = o.properties.get("text_content", "")
        image_urls = o.properties.get("original_article_image_urls", [])
        image_descriptions = o.properties.get("image_descriptions", [])

        if not isinstance(image_urls, list):
            image_urls = [image_urls]
        if not isinstance(image_descriptions, list):
            image_descriptions = [image_descriptions]

        for desc, img_url in zip(image_descriptions, image_urls):
            if is_valid_image_url(img_url):
                desc_embedding = text_embeddings_model.embed_query(desc)
                similarity = cosine_similarity([query_embedding], [desc_embedding])[0][0]
                if similarity > top_image_similarity:
                    top_image_similarity = similarity
                    top_image_url = img_url

        if url and url not in seen_urls:
            unique_articles_with_images.append({
                "content": content,
                "title": title,
                "url": url,
                "image_urls": image_urls
            })
            seen_urls.add(url)

        if len(unique_articles_with_images) >= top_k_text:
            break

    return unique_articles_with_images, top_image_url


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

            relevant_content_docs, top_image_url = retrieve_content(prompt, text_embeddings_model, weaviate_chunks_collection, top_k_text=5)

            if not relevant_content_docs:
                assistant_reply = (
                    "I couldn't find specific articles for your query in my database 💁‍♀️"
                    "Please try rephrasing your question."
                )
                st.markdown(assistant_reply)
                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                st.stop()

            context_text = ""
            unique_article_links = {}
            for doc in relevant_content_docs[:5]:
                context_text += f"Content: {doc['content']}\nTitle: {doc['title']}\n\n"
                if 'url' in doc:
                    unique_article_links[doc['url']] = doc['title']

            article_links_output = ""
            for url, title in unique_article_links.items():
                article_links_output += f"- **{title}** ([Read More]({url}))\n"

            full_prompt = f"""
### Role and Goal
You are an expert on articles, providing helpful, comprehensive, and accurate answers. Your primary goal is to provide a detailed and accurate answer to the user's question, 
*strictly based on the provided article context*. If the context does not contain enough information to fully answer, clearly state that you cannot answer based on the given information.

### Constraints
- Do NOT include any links, URLs, markdown links (e.g., [text](url)), or direct references to article titles within your main answer. These will be handled separately at the end of the response.
- Do NOT make up information. Only use the provided context.
- Do NOT answer questions unrelated to the articles or context.
- Do NOT repeat the user's question in the answer unless necessary for clarity.
- Stay neutral and factual. Do not express opinions or speculate beyond the content.

### Context from Articles (multiple chunks)
{context_text}

### Relevant Image Information
The user is also being shown one relevant image that was found for their query. 
- You may briefly acknowledge the image **only if the article context directly refers to or explains its content.**
- If the article context does NOT mention the topic of the user’s question, then you must NOT mention the image at all.
- Do NOT attempt to interpret or describe the image in any way if the topic is not covered in the article.
- In such cases, you must add at the end: "There is no relevant image available either."

### Conversation History
Use the following conversation history to preserve continuity and understand any clarifying details the user may have shared.
{chat_history_str}

### User Question
{prompt}

### Task
Provide your detailed and accurate answer based on the above information.
"""

            try:
                gemini_response = model.generate_content(full_prompt, generation_config=genai.GenerationConfig(temperature=0.1))
                assistant_reply = re.sub(r'!?\[.*?\]\(https?://[^\s)]+\)|https?://[^\s)]+', '', gemini_response.text).strip()
            except Exception as e:
                assistant_reply = f"An error occurred while generating the response: {e}"

            if top_image_url:
                assistant_reply += "\n\n---\n\nHere is a relevant image 🎥:\n"
                assistant_reply += f"![Relevant Image]({top_image_url})\n\n"

            if article_links_output:
                assistant_reply += "\n\n---\n\nFor more information, click on the links below 👀👀👀:\n" + article_links_output
            # Show retrieved context chunks for evaluation/debugging
            with st.expander("🔍 View Retrieved Context Chunks"):
                for idx, doc in enumerate(relevant_content_docs):
                    st.markdown(f"**Chunk {idx + 1}:**")
                    st.text(doc["content"])

            st.markdown(assistant_reply)
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})