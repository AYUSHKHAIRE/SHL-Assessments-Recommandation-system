from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from dotenv import load_dotenv

from models.text_processor import TextProcessor
from models.geminiembeder import GeminiEmbedder
from models.google_gemini import GoogleGemini

# ---------------------------
# App
# ---------------------------
app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0"
)

# ---------------------------
# Config
# ---------------------------

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

GG = GoogleGemini(api_key=GEMINI_API_KEY)

df = pd.read_csv("data/assessments_details.csv")

# ---------------------------
# Schemas
# ---------------------------
class RecommendRequest(BaseModel):
    query: str
    top_k: int = 10

# ---------------------------
# Endpoint
# ---------------------------
@app.post("/recommend")
def recommend(req: RecommendRequest):
    # 1. Refine query using Gemini
    prompt = f"""
    Given the query, generate a concise job description capturing skills and role.
    Query: {req.query}
    """
    refined_query = GG.generate(prompt)

    # 2. Preprocess + embed
    processed_query = " ".join(TP.process_text(refined_query))
    query_embedding = EB.get_user_query_embedding(processed_query)

    # 3. Vector search
    scores, texts, ids = EB.get_similar_chunks(
        query_embedding,
        top_k=req.top_k
    )

    # 4. Fetch rows
    selected = df[df["dataset_id"].isin(ids)].copy()
    selected["similarity_score"] = selected["dataset_id"].apply(
        lambda x: scores[ids.index(x)]
    )

    selected = selected.sort_values(
        by="similarity_score",
        ascending=False
    )

    # 5. Format response
    recommended_assessments = []
    for row in selected.itertuples():
        recommended_assessments.append({
            "url": row.link,
            "name": row.heading,
            "adaptive_support": "No",  # not available in dataset
            "description": row.desc,
            "duration": int(
                str(row.assessment_length).split()[-1]
            ),
            "remote_support": row.remote_testing,
            "test_type": [
                t.strip()
                for t in row.test_type
                .replace("Test Type:", "")
                .split(",")
            ],
            "similarity_score": round(row.similarity_score, 4)
        })

    return {
        "query": req.query,
        "recommended_assessments": recommended_assessments
    }

# uvicorn api.main:app --host 0.0.0.0 --port 8000