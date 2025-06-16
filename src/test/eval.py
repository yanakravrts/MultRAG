import pandas as pd
import plotly.graph_objects as go
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import MultiModalFaithfulness, MultiModalRelevance, LLMContextPrecisionWithoutReference, LLMContextRecall, ContextEntityRecall, ResponseRelevancy, Faithfulness
from ragas import evaluate
from dotenv import load_dotenv, find_dotenv
import os
import google.generativeai as genai
from datasets import Dataset


load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-1.5-flash"))
evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))

data_samples = {
    "question": [
        "What did Skyfire AI drone do during a traffic stop?",
        "When to use QLoRa?",
        "Comparison between Gemma and different models"
    ],
    "reference": [
        "The Skyfire AI drone was redirected to find a missing officer during a traffic stop.",
        "I couldn't find specific articles for your query in my database 💁‍♀️Please try rephrasing your question.",
        "Gemma 3 4B is comparable to Gemma 2 9B, Llama 3.1 8B, and Qwen2.5-7B, and slightly trails Microsoft Phi-4 Mini except on MATH."
    ],
    "answer": [
        "The Skyfire AI drone was rerouted to help locate a police officer who had stopped a suspicious vehicle and hadn't been heard from since. The officer had stopped at a complex cloverleaf intersection of two major highways, and the drone's assistance was requested because dispatch was unsure of the officer's exact location. The drone was returning to base after a false alarm when it was redirected.",
        "I couldn't find specific articles for your query in my database 💁‍♀️Please try rephrasing your question.",
        "Gemma 3 4B performs similarly to Gemma 2 9B, Llama 3.1 8B, and Qwen2.5-7B, but slightly trails Microsoft Phi-4 Mini (also a 4-billion parameter model) except in MATH performance, based on Microsoft's testing. Gemma 3 12B surpasses Gemma 2 27B and shows comparable performance to Gemini 1.5 Flash (according to TIGER-Lab's tests) and Anthropic Claude 3.5 Haiku (according to Anthropic's tests). It outperforms these larger, proprietary models on MATH. There is no relevant image available either."
    ],
    "contexts": [
        ["In January, a Skyfire AI drone was returning to base after responding to a false alarm when the police dispatcher asked us to reroute it to help locate a patrol officer. The officer had radioed a few minutes earlier that he had pulled over a suspicious vehicle and had not been heard from since. The officer had stopped where two major highways intersect in a complex cloverleaf, and dispatch was unsure exactly where they were located."],
        [],
        ["Gemma 3 4B achieves roughly comparable performance to Gemma 2 9B, Llama 3.1 8B, and Qwen2.5-7B. It’s slightly behind Microsoft Phi-4 Mini (also 4 billion parameters), except on MATH, according to that company’s tests.Gemma 3 12B improves on Gemma 2 27B and compares to Gemini 1.5 Flash (in TIGER-Lab’s tests) and Anthropic Claude 3.5 Haiku (in that developer’s tests). It outperforms the larger, proprietary models on MATH."]
    ],
    "image": [
        ["https://dl-staging-website.ghost.io/content/images/2025/02/unnamed--52-.png"],
        [],
        ["https://dl-staging-website.ghost.io/content/images/2025/02/unnamed--52-.jpg"]
    ]
}
dataset = Dataset.from_dict(data_samples)

score = evaluate(
    dataset, metrics=[
        MultiModalFaithfulness(),
        MultiModalRelevance(),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        ContextEntityRecall(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm,embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm)
          ],
      llm=evaluator_llm,
      embeddings=evaluator_embeddings
)

df = score.to_pandas()


fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df[col] for col in df.columns],
               fill_color='lavender',
               align='left'))
])


fig.update_layout(title_text="Evaluation")

fig.write_html("ragas_evaluation_table.html")

print("Інтерактивну таблицю збережено у файл ragas_evaluation_table.html")