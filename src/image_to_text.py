import os
import json
import base64
import requests
from io import BytesIO
import mimetypes 
import time      
from pathlib import Path

from tqdm import tqdm
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document


FREE_TIER_RPM = 15
REQUEST_DELAY_SECONDS = (60 / FREE_TIER_RPM) + 0.5 

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_image_base64_from_url(image_url: str) -> tuple[str | None, str | None]:
    """
    Downloads an image from a URL and returns its base64 encoding and MIME type.
    """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status() 
        content_type = response.headers.get('Content-Type')
        if not content_type:
            content_type, _ = mimetypes.guess_type(image_url)
            if not content_type:
                print(f"WARNING: Could not determine MIME type for {image_url}. Defaulting to 'application/octet-stream'.")
                content_type = "application/octet-stream"
        
        if 'image' not in content_type and image_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
             if image_url.lower().endswith(('.png')):
                 content_type = 'image/png'
             elif image_url.lower().endswith(('.jpg', '.jpeg')):
                 content_type = 'image/jpeg'
             elif image_url.lower().endswith(('.gif')):
                 content_type = 'image/gif'
             elif image_url.lower().endswith(('.webp')):
                 content_type = 'image/webp'


        img_bytes = BytesIO(response.content)
        base64_string = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return base64_string, content_type
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download or access image from URL {image_url}: {e}")
        return None, None
    except Exception as e:
        print(f"ERROR: Unknown error processing image from URL {image_url}: {e}")
        return None, None


def create_image_descriptions(docs_with_image_urls: list[Document]) -> list[dict]:
    """
    Generates descriptions for images associated with LangChain Documents using Gemini.
    Downloads images on the fly if only URLs are provided.

    Args:
        docs_with_image_urls (list[Document]): A list of LangChain Document objects,
                                            where each Document represents an article
                                            and has 'image_urls' in its metadata.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains
                    the generated image description and its associated metadata.
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
    all_generated_descriptions = []

    if not docs_with_image_urls:
        print("No article documents with image URLs provided for description generation.")
        return []

    for article_doc in tqdm(docs_with_image_urls, desc="Processing articles for image descriptions"):
        article_title = article_doc.metadata.get('title', 'Untitled')
        article_url = article_doc.metadata.get('url', 'Unknown')
        image_urls_in_article = article_doc.metadata.get('image_urls', [])

        if not image_urls_in_article:
            print(f"INFO: Article '{article_title}' has no image URLs. Skipping.")
            continue

        for idx, img_url in enumerate(image_urls_in_article):
            print(f"\nDEBUG: Attempting to process image {idx+1}/{len(image_urls_in_article)} from article '{article_title}' (URL: {img_url})")

            img_base64, mime_type = get_image_base64_from_url(img_url)

            if img_base64 and mime_type:
                image_url_data = {"url": f"data:{mime_type};base64,{img_base64}"}

                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": """
                                   Describe only the factual content visible in the image:

                                   1. If decorative/non-informational: output '<---image--->'

                                   2. For content images: 
                                   - General Images: List visible objects, text, and measurable attributes
                                   - Charts/Infographics: State all numerical numerical values and labels present
                                   - Gif: Describe the sequence of actions or changes observed frame by frame, or summarize the overall animation.

                                   Rules:
                                   * Include only directly observable information
                                   * Use original numbers and text without modification
                                   * Avoid any interpretation or analysis
                                   * Preserve all labels and measurements exactly as shown
                                   * Ensure the description is comprehensive, capturing all key visual elements
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": image_url_data, 
                        },
                    ]
                )
                
                try:
                    response = model.invoke([message])
                    print(f"DEBUG: Gemini response for image {idx+1} (Article: '{article_title}'): {response.content[:100]}...")
                    
                    description_entry = {
                        "description": response.content,
                        "article_title": article_title,
                        "article_url": article_url,
                        "image_index_in_article": idx,
                        "original_image_url": img_url,
                        "mime_type_used": mime_type,
                        "type": "image_description"
                    }
                    all_generated_descriptions.append(description_entry)

                except Exception as e:
                    print(f"ERROR: Failed to generate description for image {img_url} (Article: '{article_title}'): {e}")
                
                print(f"DEBUG: Waiting for {REQUEST_DELAY_SECONDS} seconds before next API call...")
                time.sleep(REQUEST_DELAY_SECONDS)
            
    return all_generated_descriptions

if __name__ == "__main__":
    article_data_input_path = RAW_DATA_DIR / 'article_data.json'
    output_json_path = PROCESSED_DATA_DIR / 'image_descriptions.json' 

    try:
        with open(article_data_input_path, 'r', encoding='utf-8') as f:
            articles_data_from_json = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {article_data_input_path}")
        exit()
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON in file {article_data_input_path}")
        exit()
    
    docs_for_processing = []
    for article_dict in tqdm(articles_data_from_json, desc="Preparing documents for processing"):
        if 'image_urls' in article_dict and article_dict['image_urls']:
            doc = Document(
                page_content="",
                metadata={
                    'title': article_dict.get('title', 'Untitled'),
                    'url': article_dict.get('url', 'Unknown'),
                    'image_urls': article_dict['image_urls'] 
                }
            )
            docs_for_processing.append(doc)

    if not docs_for_processing:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        exit()
    
    print(f"Found {len(docs_for_processing)} articles with image URLs")
    
    generated_descriptions = create_image_descriptions(docs_for_processing)

    if generated_descriptions:
        print(f"\nGenerated {len(generated_descriptions)} image descriptions. Saving to {output_json_path}...")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(generated_descriptions, f, ensure_ascii=False, indent=4)
    
        except Exception as e:
            print(f"ERROR: Failed to save descriptions to file {output_json_path}: {e}")
    else:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)