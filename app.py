import streamlit as st 
import pandas as pd
from models.text_processor import TextProcessor
from models.geminiembeder import GeminiEmbedder
from models.google_gemini import GoogleGemini
from dotenv import load_dotenv
import os

# configurations

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST_URL = os.getenv("QDRANT_HOST_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

cloud = True  # Set to False for local testing

if cloud:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

TP = TextProcessor()

EB = GeminiEmbedder(
    collection_name="collection_1",
    model_name="models/text-embedding-004",
    gemini_api_key=GEMINI_API_KEY,
    qdrant_url=QDRANT_HOST_URL,
    qdrant_api_key=QDRANT_API_KEY,
)

GG = GoogleGemini(api_key=GEMINI_API_KEY)

# the code handling user input and displaying results
progress_bar = st.progress(0)
df = pd.read_csv("data/assessments_details.csv")

st.title("SHL Assessment Recommender System")
st.markdown(
    """
            Prepare for your next role and land a brand new job with [SHL.com](https://www.shl.com)!

            Our recommender system helps you find the most relevant assessments based on your job description.

            Simply enter your job description query below and get personalized assessment recommendations.
    """)

query = st.text_area(
    "Enter your job description query:",
     value="I want to hire a Senior Data Analyst with 5 years of experience and expertise in SQL, Excel and Python. The assessment can be 1-2 hour long",
    height=200)

if st.button("Get Recommendations"):
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
    with st.spinner("Generating recommendations..."):
        progress_bar.progress(10)
        response = GG.generate(prompt)
        progress_bar.progress(20)
        processed_query = " ".join(TP.process_text(response))
        progress_bar.progress(40)
        query_embedding = EB.get_user_query_embedding(processed_query)
        progress_bar.progress(60)
        top_scores, top_texts, top_ids = EB.get_similar_chunks(query_embedding, top_k=30)
        st.subheader("Top Recommendations:")
        progress_bar.progress(80)
        selected_rows = df[df["dataset_id"].isin(top_ids)].copy()
        selected_rows["similarity_score"] = selected_rows["dataset_id"].apply(
            lambda x: top_scores[top_ids.index(x)]
        )
        selected_rows = selected_rows.sort_values(by="similarity_score", ascending=False)
        # selected_rows.drop_duplicates(subset=["link"], inplace=True)
        for row in selected_rows.itertuples():
            st.markdown(f"### [{row.heading}]({row.link})")
            st.markdown(f"**Type:** {row.test_type} | **Duration:** {row.assessment_length} minutes")
            st.markdown(f"**Description:** {row.desc}")
            st.markdown(f"**Similarity Score:** {row.similarity_score:.4f}")
            st.markdown("---")
        progress_bar.progress(100)
        st.success("Recommendations generated!")

# build sidebar filters

job_levels = df["job_levels"].unique().tolist()
splited_job_levels = [level.split(",") for level in job_levels]
all_levels = [level.strip() for sublist in splited_job_levels for level in sublist]
unique_job_levels = list(set(all_levels))

all_test_types = df["test_type"].replace("Test Type:", "").unique().tolist()
splited_test_types = [level.split(":")[-1] for level in all_test_types]
all_test_types = [level.strip() for sublist in splited_test_types for level in sublist]
unique_test_types = list(set(all_test_types))

languages = df["languages"].unique().tolist()
splited_languages = [lang.split(",") for lang in languages]
all_languages = [lang.strip() for sublist in splited_languages for lang in sublist]
language_options = list(set(all_languages))

times = df["assessment_length"].unique().tolist()
processed_times = [(time.split("=")[-1]) for time in times]
processed_times = [time.strip().replace(" ", "") for time in processed_times]
time_options = sorted(list(set(processed_times)))

Filtered_df = df.copy()

# if sidebar filters are applied
def filter_dataframe(df, assessment_type, job_level, language, max_time):
    filtered_df = df.copy()

    if assessment_type:
        filtered_df = filtered_df[filtered_df["test_type"].apply(
            lambda x: any(at in x for at in assessment_type)
        )]

    if job_level:
        filtered_df = filtered_df[filtered_df["job_levels"].apply(
            lambda x: any(jl in x for jl in job_level)
        )]

    if language:
        filtered_df = filtered_df[filtered_df["languages"].apply(
            lambda x: any(lang in x for lang in language)
        )]

    if max_time:
        filtered_df = filtered_df[filtered_df["assessment_length"].apply(
            lambda x: any(mt in x for mt in max_time)
        )]

    return filtered_df

st.dataframe(Filtered_df)

with st.sidebar:
    st.header("Apply filters")

    assessment_type = st.multiselect(
        "Select Assessment Type:",
        options=unique_test_types,
        default=None
    )

    job_level = st.multiselect(
        "Select Job Level:",
        options=unique_job_levels,
        default=None
    )

    language = st.multiselect(
        "Select Language:",
        options=language_options,
        default=None
    )

    max_time = st.multiselect(
        "Select Maximum Assessment Duration (minutes):",
        options=time_options,
        default=None
    )