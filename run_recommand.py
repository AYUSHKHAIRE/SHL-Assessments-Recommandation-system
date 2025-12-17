import pandas as pd
from models.text_processor import TextProcessor
from models.geminiembeder import GeminiEmbedder
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST_URL = os.getenv("QDRANT_HOST_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TP = TextProcessor()

EB = GeminiEmbedder(
    collection_name="collection_1",
    model_name="models/text-embedding-004",
    gemini_api_key=GEMINI_API_KEY,
    qdrant_url=QDRANT_HOST_URL,
    qdrant_api_key=QDRANT_API_KEY,
)

df = pd.read_csv("data/assessments_details.csv")

job_descriptions = df["desc"].tolist()
ass_ids = df["dataset_id"].tolist()

query = (
    "I want to hire a Senior Data Analyst with 5 years of experience "
    "and expertise in SQL, Excel and Python. The assessment can be 1-2 hour long"
)

processed_query = " ".join(TP.process_text(query))
query_embedding = EB.get_user_query_embedding(processed_query)

top_scores, top_texts, top_ids = EB.get_similar_chunks(query_embedding, top_k=15)

for score, text, dataset_id in zip(top_scores, top_texts, top_ids):
    print(f"Score: {score:.4f}, Job Description: {text[:500]}..., Dataset ID: {dataset_id}")