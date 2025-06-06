# MultRAG: Advanced AI Chatbot for DeepLearning.AI The Batch Articles

## Overview

MultRAG (Multi-modal Retrieval Augmented Generation) is an AI chatbot designed to answer questions specifically about articles from DeepLearning.AI's "The Batch" newsletter. It leverages a combination of text embeddings (for content retrieval) and CLIP embeddings (for image retrieval) to provide comprehensive and contextually relevant answers. The chatbot is built using Streamlit for the user interface and integrates with Google's Gemini-1.5-Flash model for natural language generation.

## Features

- **Intelligent Text Retrieval:** Uses FAISS vector store with BAAI/bge-base-en-v1.5 embeddings to find the most relevant article snippets based on user queries.
- **Image Retrieval:** Utilizes CLIP embeddings to find and display images semantically related to the user's question.
- **Context-Aware Responses:** Generates answers using Google's Gemini-1.5-Flash model, informed by retrieved text context and chat history.
- **Structured Output:** Presents answers, relevant images, and source links in a clear, organized manner.
- **Dynamic Data Collection:** Includes scripts to scrape and process articles from "The Batch" website.

## Setup Instructions

### 1. Clone the Repository


git clone [https://github.com/yanakravrts/MultRAG.git](https://github.com/yanakravrts/MultRAG.git)


### 2. Set Up Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

python3 -m venv venv
source venv/bin/activate  

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Configure Environment Variables

##### .env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

### 5. Data Collection and Embedding Generation

#### a. Collect Raw Article Data

This script scrapes article content and image URLs from the DeepLearning.AI "The Batch" website and saves them to data/raw/article_data.json.

python3 fetch_articles.py

#### b. Generate Text Embeddings

python3 text_embeddings.py

#### c. Generate Media Embeddings

python3 media_embeddings.py

### 6. Run the Application

streamlit run app.py


## Usage
Type your questions related to DeepLearning.AI "The Batch" articles into the input field.
The chatbot will retrieve relevant text snippets and images, and generate an answer.
Sources (article links) and a relevant image will be displayed below the generated answer.

## Evaluation

### Text Evaluation
- Precision@3 = 0.3333  
- Recall = 1.0000  
- Retrieved Text URLs:  
  - [https://www.deeplearning.ai/the-batch/issue-289/](https://www.deeplearning.ai/the-batch/issue-289/)  
- Relevant Text URLs (Ground Truth):  
  - [https://www.deeplearning.ai/the-batch/issue-289/](https://www.deeplearning.ai/the-batch/issue-289/)

---

- Precision@3 = 0.3333  
- Recall = 1.0000  
- Retrieved Text URLs:  
  - [https://www.deeplearning.ai/the-batch/issue-290/](https://www.deeplearning.ai/the-batch/issue-290/)  
  - [https://www.deeplearning.ai/the-batch/issue-291/](https://www.deeplearning.ai/the-batch/issue-291/)  
- Relevant Text URLs (Ground Truth):  
  - [https://www.deeplearning.ai/the-batch/issue-290/](https://www.deeplearning.ai/the-batch/issue-290/)

---

- Precision@3 = 0.3333  
- Recall = 1.0000  
- Retrieved Text URLs:  
  - [https://www.deeplearning.ai/the-batch/issue-294/](https://www.deeplearning.ai/the-batch/issue-294/)  
  - [https://www.deeplearning.ai/the-batch/issue-293/](https://www.deeplearning.ai/the-batch/issue-293/)  
- Relevant Text URLs (Ground Truth):  
  - [https://www.deeplearning.ai/the-batch/issue-294/](https://www.deeplearning.ai/the-batch/issue-294/)

---

### Image Evaluation
- Precision@1 = 1.0000  
- Retrieved Image URL:  
  - ![Image 1](https://dl-staging-website.ghost.io/content/images/2025/02/unnamed--52-.png)  
- Relevant Image URLs (Ground Truth):  
  - [https://dl-staging-website.ghost.io/content/images/2025/02/unnamed--52-.png](https://dl-staging-website.ghost.io/content/images/2025/02/unnamed--52-.png)

---

- Precision@1 = 1.0000  
- Retrieved Image URL:  
  - ![Image 2](https://dl-staging-website.ghost.io/content/images/2025/02/unnamed--52-.jpg)  
- Relevant Image URLs (Ground Truth):  
  - [https://dl-staging-website.ghost.io/content/images/2025/02/unnamed--52-.jpg](https://dl-staging-website.ghost.io/content/images/2025/02/unnamed--52-.jpg)

---

- Precision@1 = 0.0000  
- Retrieved Image URL:  
  - ![Image 3](https://dl-staging-website.ghost.io/content/images/2025/04/unnamed--56-.gif)  
- Relevant Image URLs (Ground Truth):  
  - [https://dl-staging-website.ghost.io/content/images/2025/03/unnamed--56-.jpg](https://dl-staging-website.ghost.io/content/images/2025/03/unnamed--56-.jpg)

---

**Note:** Low Precision happens because of duplicates.
