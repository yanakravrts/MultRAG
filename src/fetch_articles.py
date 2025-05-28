import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path 


batch_url = "https://www.deeplearning.ai/the-batch/"
base_url = "https://www.deeplearning.ai"

def get_all_articles_links():
    """
    Retrieves all unique article links from the main The Batch page.

    Returns:
        list: A list of full URLs to article issues.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    response = requests.get(batch_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    links = []
    for a in soup.find_all("a", href=True):
        href = a['href']
        if href.startswith("/the-batch/issue-"):
            full_url = base_url + href
            links.append(full_url)

    return list(set(links))  

def clean_image_urls(image_urls):
    """
    Cleans and normalizes a list of image URLs.

    Args:
        image_urls (list): Raw image URLs extracted from the article.

    Returns:
        list: A cleaned list of direct image URLs.
    """
    clean_urls = []
    for url in image_urls:
        if url.startswith("data:image"):
            continue  # Skip base64 images
        
        if url.startswith("https://www.deeplearning.ai/_next/image/?url="):
            parsed_url = urlparse(url)
            query = parse_qs(parsed_url.query)
            if "url" in query:
                decoded_path = unquote(query["url"][0])
                if decoded_path.startswith("/"):
                    full_url = base_url + decoded_path
                else:
                    full_url = decoded_path
                clean_urls.append(full_url)
            else:
                clean_urls.append(url)
        else:
            clean_urls.append(url)
    
    return clean_urls

def parse_article(url):
    """
    Parses a single article from The Batch.

    Args:
        url (str): URL of the article.

    Returns:
        dict or None: A dictionary containing article title, date, URL, content, and image URLs.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")

        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No Title"
        date = soup.find("time").get("datetime") if soup.find("time") else None

        content_divs = soup.select("article div")
        content = "\n".join([div.get_text(strip=True) for div in content_divs])

        imgs = soup.select("article img")
        image_urls = []
        for img in imgs:
            src = img.get("src")
            if src:
                if src.startswith("/"):
                    src = base_url + src
                image_urls.append(src)

        image_urls = clean_image_urls(image_urls)

        if not image_urls:
            image_urls = None

        return {
            "title": title,
            "date": date,
            "url": url,
            "content": content,
            "image_urls": image_urls,
        }
    except Exception as e:
        print(f"[!] Failed to parse {url}: {e}")
        return None

def collect_articles():
    """
    Main function that:
    - Collects all article links
    - Parses each article
    - Saves the collected data to a JSON file
    """
    links = get_all_articles_links()
    data = []

    for link in tqdm(links, desc="Collecting articles"):
        article = parse_article(link)
        if article:
            data.append(article)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
 
    output_path = output_dir / "article_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Collected articles saved to {output_path}")
    print(f"Number of articles collected: {len(data)}")

if __name__ == "__main__":
    collect_articles()
