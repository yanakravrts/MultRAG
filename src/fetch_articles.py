import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from urllib.parse import urlparse, parse_qs, unquote, urljoin
from pathlib import Path 


batch_url = "https://www.deeplearning.ai/the-batch/"
base_url = "https://www.deeplearning.ai"
TOTAL_BATCH_PAGES = 2


def get_all_articles_links():
    """
    Retrieves all unique article links from all specified pages of The Batch.

    Returns:
        list: A list of full URLs to article issues.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    
    all_links = set()  

    for page_num in tqdm(range(1, TOTAL_BATCH_PAGES + 1), desc="Scanning pages for links"):
        if page_num == 1:
            current_page_url = batch_url 
        else:
            current_page_url = f"{batch_url}page/{page_num}/" 

        print(f"Scraping page {page_num}: {current_page_url}")
        
        try:
            response = requests.get(current_page_url, headers=headers, timeout=15) 
            response.raise_for_status() 
            soup = BeautifulSoup(response.content, 'html.parser')

            for a in soup.find_all("a", href=True):
                href = a['href']
                if (
                    href.startswith("/the-batch/") and
                    not href.startswith("/the-batch/tag") and 
                    href != "/the-batch/" and 
                    not href.startswith("/the-batch/page/") 
                ):
                    full_url = urljoin(base_url, href) 
                    all_links.add(full_url)

        except requests.exceptions.RequestException as e:
            print(f"[!] Error fetching {current_page_url}: {e}")
            continue 

    return list(all_links)


def clean_image_urls(image_urls):
    """
    Cleans and normalizes a list of image URLs.
    """
    clean_urls = []
    for url in image_urls:
        if url.startswith("data:image"):
            continue  
        
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
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }
        r = requests.get(url, headers=headers, timeout=15) 
        r.raise_for_status() 
        soup = BeautifulSoup(r.content, "html.parser")

        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No Title"
        date = soup.find("time").get("datetime") if soup.find("time") else None
        content_elements = soup.select("div.prose.max-w-none.md\\:prose-lg.text-base.md\\:text-lg p, div.prose.max-w-none.md\\:prose-lg.text-base.md\\:text-lg h2, div.prose.max-w-none.md\\:prose-lg.text-base.md\\:text-lg li")
        if not content_elements: 
            content_elements = soup.select("article p, article h2, article li")
        
        content = "\n".join([el.get_text(strip=True) for el in content_elements])

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
    except requests.exceptions.RequestException as e:
        print(f"[!] HTTP Error parsing {url}: {e}")
        return None
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

    valid_links = [link for link in links if link is not None]

    for link in tqdm(valid_links, desc="Collecting articles"):
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