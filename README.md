# MultRAG: Advanced AI Chatbot for DeepLearning.AI The Batch Articles

## Overview

MultRAG (Multi-modal Retrieval Augmented Generation) is an AI chatbot designed to answer questions specifically about articles from DeepLearning.AI's "The Batch" newsletter. It leverages a combination of text embeddings (for content retrieval) and CLIP embeddings (for image retrieval) to provide comprehensive and contextually relevant answers. The chatbot is built using Streamlit for the user interface and integrates with Google's Gemini-1.5-Flash model for natural language generation.

## Features

- **Intelligent Text Retrieval:** Uses FAISS vector store with BAAI/bge-base-en-v1.5 embeddings to find the most relevant article snippets based on user queries.
- **Image Retrieval:** Utilizes CLIP embeddings to find and display images semantically related to the user's question.
- **Context-Aware Responses:** Generates answers using Google's Gemini-1.5-Flash model, informed by retrieved text context and chat history.
- **Structured Output:** Presents answers, relevant images, and source links in a clear, organized manner.
- **Dynamic Data Collection:** Includes scripts to scrape and process articles from "The Batch" website.

## Project Structure


MultRAG/
├── data/
│ ├── raw/
│ │ └── article_data.json # Raw scraped article data
│ └── embeddings/
│ ├── text_embeddings/ # FAISS index for text embeddings
│ ├── media_embeddings.faiss # FAISS index for image embeddings
│ └── media_urls.json # URLs corresponding to media_embeddings
├── app.py # Main Streamlit application
├── fetch_articles.py # Script to scrape and process articles
├── text_embeddings.py # Script to generate and save text embeddings
├── media_embeddings.py # Script to generate and save image embeddings
├── requirements.txt # Python dependencies
├── .env.example # Example for environment variables
├── .gitignore # Specifies files/directories to ignore by Git
└── README.md # This documentation file

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
