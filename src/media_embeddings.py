import json
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
from pathlib import Path 


json_path = "data/raw/article_data.json" 

with open(json_path, "r", encoding="utf-8") as f:
    articles = json.load(f)

def is_valid_image_url(url):
    """
    Check if the provided URL is a valid image URL for processing.
    - Reject data URLs for SVG and GIF images.
    - Accept only URLs starting with http:// or https://.
    """
    if not url:
        return False
    url = url.lower()
    if url.startswith("data:image/svg+xml") or url.startswith("data:image/gif;base64"):
        return False
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    return True

image_urls = []
for article in articles:
    if "image_url" in article:
        if is_valid_image_url(article["image_url"]):
            image_urls.append(article["image_url"])
    if "image_urls" in article:
        for url in article["image_urls"]:
            if is_valid_image_url(url):
                image_urls.append(url)

image_urls = list(set(image_urls))

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name, use_fast=True) 

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def get_image_embedding(image_url):
    """
    Download an image from the given URL and compute its CLIP embedding.
    Returns the normalized embedding as a NumPy array, or None if failed.
    """
    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
    except Exception as e:
        return None


embeddings_list = []
valid_urls = []

for i, url in enumerate(image_urls):
    emb = get_image_embedding(url)
    if emb is not None:
        embeddings_list.append(emb)
        valid_urls.append(url)


if not embeddings_list:
    print("No valid image embeddings generated")
else:
    embeddings_matrix = np.vstack(embeddings_list).astype('float32')
    dimension = embeddings_matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(output_dir / "media_embeddings.faiss"))

    with open(output_dir / "media_urls.json", "w") as f:
        json.dump(valid_urls, f, indent=2)
