import pandas as pd
from models.text_processor import TextProcessor
from models.geminiembeder import GeminiEmbedder
from models.google_gemini import GoogleGemini
from evaluation.evaluation_engine import get_recall_K
from dotenv import load_dotenv
import os
from tqdm import tqdm
import time
from tqdm import tqdm
import numpy as np

# configurations

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

# data 
train_labeled_df = pd.read_excel("data/eval_datasets/train-labeled-SHL.xlsx",engine='openpyxl')
test_df = pd.read_excel("data/eval_datasets/test-SHL.xlsx",engine='openpyxl')
df = pd.read_csv("data/assessments_details.csv")

def get_recommandation_links(query):
    prompt = f"""
    Given the query , generate a concise job description that captures the key skills and requirements mentioned in the query.
    Query: {query}

    Instructions:
    - Focus on extracting relevant skills, experience, and job role.
    - Keep the description concise and to the point.
    - Avoid adding any information not present in the query.
    - The output should be a single paragraph.
    - Dont use json or markdown format.
    """
    response = GG.generate(prompt)
    processed_query = " ".join(TP.process_text(response))
    query_embedding = EB.get_user_query_embedding(processed_query)
    top_scores, top_texts, top_ids = EB.get_similar_chunks(query_embedding, top_k=10)
    selected_rows = df[df["dataset_id"].isin(top_ids)].copy()
    selected_rows["similarity_score"] = selected_rows["dataset_id"].apply(
        lambda x: top_scores[top_ids.index(x)]
    )
    selected_rows = selected_rows.sort_values(by="similarity_score", ascending=False)
    return {
        "query": query,
        "recommended_assessments": selected_rows["link"].tolist()
    }

# query = "I want to hire a Senior java developer with 5 years of experience and expertise in SQL, Excel and Python. The assessment can be 1-2 hour long"
# results = get_recommandation_links(query)
# print("Query:", results["query"])
# print("Recommended Assessments Links:")
# for link in results["recommended_assessments"]:
#     print(link)

# ------------------------------------------
# predict on the trained label data queries for evaluation
# ------------------------------------------

# print("train labeled columns:", train_labeled_df.columns)

# train_labeled_queries = train_labeled_df["Query"].unique().tolist()
# print(len(train_labeled_queries), "unique queries found in train labeled data")

# trained_queries_results = {
#     "query": [],
#     "recommended_assessments": []
# }

# for query in tqdm(train_labeled_queries,desc="Processing train labeled queries"):
#     results = get_recommandation_links(query)
#     for r in results["recommended_assessments"]:
#         trained_queries_results["query"].append(results["query"])
#         trained_queries_results["recommended_assessments"].append(r)
#         time.sleep(1)

# trained_queries_results_df = pd.DataFrame(trained_queries_results)
# trained_queries_results_df.to_csv("data/eval_results/trained_queries_recommendations.csv")

# ------------------------------------------
# predict on the test data queries for evaluation
# ------------------------------------------

# print("test columns:", test_df.columns)

# test_queries = test_df["Query"].unique().tolist()
# print(len(test_queries), "unique queries found in test labeled data")

# test_queries_results = {
#     "query": [],
#     "recommended_assessments": []
# }

# for query in tqdm(test_queries,desc="Processing test queries"):
#     results = get_recommandation_links(query)
#     for r in results["recommended_assessments"]:
#         test_queries_results["query"].append(results["query"])
#         test_queries_results["recommended_assessments"].append(r)
#         time.sleep(1)

# test_queries_results_df = pd.DataFrame(test_queries_results)
# test_queries_results_df.to_csv("data/eval_results/test_queries_recommendations.csv")

# ---------------------------------------
# get recall K
# ---------------------------------------

test_get_recall = get_recall_K(
        4,10
)
print(test_get_recall)

"""
Assumptions for the recall calculation:

in the train set there are queries and assessment links .
so let refer them it like :

number_of_relevent_assessments_in_top_k : 
    common assessments between 
    - each query result by my recommandation engine ,
    - and the original assessments given in train set
    for every query

total_relevent_assessments_for_the_query :
    the original assessments given in train set
    for every query
"""

# calculate recall for every query

query_recall_k = {}

trained_queries_recommendations_df = \
    pd.read_csv("data/eval_results/trained_queries_recommendations.csv")

unique_queries = train_labeled_df["Query"].unique().tolist()
for query in tqdm(unique_queries,desc = "Computing recall"):
    # print(train_labeled_df.columns)
    # print(trained_queries_recommendations_df.columns)
    original_assessment_links = \
        train_labeled_df[train_labeled_df["Query"] == query]["Assessment_url"].tolist()
    original_assessment_links = [ l.replace("/solutions","") for l in original_assessment_links ]
    recommanded_assessment_links = \
        trained_queries_recommendations_df[trained_queries_recommendations_df["query"] == query]["Assessment_url"].tolist()
    common_queries_between_train_and_predict = \
        list(
            set(original_assessment_links).intersection(
                set(recommanded_assessment_links)
            )
        )
    # print(recommanded_assessment_links)
    # print(common_queries_between_train_and_predict)
    # print(original_assessment_links)
    no_of_rl_ass_tpk = len(common_queries_between_train_and_predict)
    tot_rl_ass = len(original_assessment_links)
    recall_tpk = get_recall_K(
        no_of_rl_ass_tpk,
        tot_rl_ass
    )
    query_recall_k[query] = recall_tpk

# ------------------------------
# get mean recall @k 
# ------------------------------

# convert to list, ignore empty queries safely
recall_values = [
    v for v in query_recall_k.values()
    if v is not None and not np.isnan(v)
]

mean_recall_k = np.mean(recall_values)

print(f"Mean Recall@K: {mean_recall_k:.4f}")