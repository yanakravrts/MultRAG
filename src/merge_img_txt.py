import json

def load_json_file(filepath):
    """
    Loads a JSON file from the given filepath.
    Args:
        filepath (str): The path to the JSON file.
    Returns:
        list or dict: The loaded JSON data, or None if an error occurs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None


def merge_article_and_image_data(article_data, image_descriptions):
    """
    Merges article data with corresponding image descriptions.
    Args:
        article_data (list): A list of dictionaries, where each dictionary represents an article.
        image_descriptions (list): A list of dictionaries, where each dictionary represents an image description.
    Returns:
        list: A list of merged article dictionaries, each including a 'image_descriptions' key.
    """

    merged_data = []
    image_map = {}
    for img_desc in image_descriptions:
        article_url = img_desc.get("article_url")
        if article_url:
            if article_url not in image_map:
                image_map[article_url] = []
            image_map[article_url].append(img_desc.get("description", ""))

    for article in article_data:
        article_url = article.get("url")
        article_with_images = article.copy()
        
        article_with_images["image_descriptions"] = image_map.get(article_url, [])
        
        merged_data.append(article_with_images)

    return merged_data


def save_json_file(data, filepath):
    """
    Saves data to a JSON file.
    Args:
        data (list or dict): The data to save.
        filepath (str): The path where the JSON file will be saved.
    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to {filepath}")
        return True
    except IOError as e:
        print(f"Error: Could not save data to {filepath}. {e}")
        return False


if __name__ == "__main__":
    article_data_path = 'data/raw/article_data.json'
    image_descriptions_path = 'data/processed/image_descriptions.json'
    output_merged_filepath = 'data/processed/merged_articles_with_images.json'

    article_data = load_json_file(article_data_path)
    image_descriptions = load_json_file(image_descriptions_path)

    if article_data is None or image_descriptions is None:
        print("Exiting due to data loading errors.")
    else:
        print("Merging article and image data...")
        merged_articles = merge_article_and_image_data(article_data, image_descriptions)
        print(f"Successfully merged data for {len(merged_articles)} articles.")

        if merged_articles:
            print(f"\nSaving merged data to {output_merged_filepath}...")
            save_json_file(merged_articles, output_merged_filepath)
        else:
            print("No merged data to save.")