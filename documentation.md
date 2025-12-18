## 5. Documentation  

### `app.py`

- Purpose: Streamlit app that recommends SHL assessments based on a job description query.
- Key libraries: `streamlit`, `pandas`, `dotenv`; local modules: `models.text_processor.TextProcessor`, `models.geminiembeder.GeminiEmbedder`, `models.google_gemini.GoogleGemini`.
- Data source: reads `data/assessments_details.csv`.
- Environment variables / secrets:
  - `QDRANT_API_KEY`, `QDRANT_HOST_URL`, `GEMINI_API_KEY` (from `.env` or `st.secrets` when `cloud=True`).
- Main objects:
  - `TP` — `TextProcessor()` for preprocessing.
  - `EB` — `GeminiEmbedder(...)` for embeddings and retrieving similar chunks from Qdrant.
  - `GG` — `GoogleGemini(api_key=...)` for prompt-based query condensation.
- Main UI flow (on "Get Recommendations"):
  - Condense the user query with `GG.generate(prompt)`.
  - Preprocess text with `TP.process_text(...)`.
  - Create an embedding via `EB.get_user_query_embedding(...)`.
  - Retrieve similar chunks using `EB.get_similar_chunks(..., top_k=10)`.
  - Map `top_ids` to rows in the CSV, compute `similarity_score`, and display results (`heading`, `link`, `test_type`, `assessment_length`, `desc`).
- Sidebar filters:
  - Builds options for assessment type, job level, language, and max duration; `filter_dataframe(...)` applies selected filters to the dataframe.
- UI/feedback elements: uses `st.progress`, `st.spinner`, `st.success`, and `st.dataframe`.
- Run command: `streamlit run app.py`
- Notes / caveats:
  - `cloud` flag toggles use of `st.secrets`; ensure secrets are configured when enabled.
  - Potential variable mismatch: `QDRANT_URL` is set from secrets but `EB` is initialized with `qdrant_url=QDRANT_HOST_URL` — verify which variable is intended.

![](/assets/demo.gif)

### `assign_ids.py`

- Purpose: Assigns unique `dataset_id` values to rows in a CSV file (used for mapping dataset records to embedding/vector IDs).
- Main libraries: `pandas`, `uuid.uuid4`.
- Primary function:
  - `assign_dataset_ids(filepath: str) -> list`:
    - Loads CSV at `filepath`.
    - If `dataset_id` column is missing, creates it and fills with UUIDs.
    - If `dataset_id` exists, assigns new UUIDs only to empty/NaN entries.
    - Saves the updated dataframe back to `filepath`.
    - Returns the list of `dataset_id` values.
- Default behavior: The module calls `assign_dataset_ids('data/assessments_details.csv')` when executed as a script.
- Notes / caveats:
  - Running the script overwrites `data/assessments_details.csv` in-place—back up the file if needed before running.
  - IDs are generated with `uuid4()` and returned as strings.

### `run_evaluation.py`

- **Purpose**:  
  Runs the end-to-end evaluation pipeline for the recommendation system by:
  - Generating recommendations for labeled queries
  - Comparing them against ground-truth assessments
  - Computing Recall@K and Mean Recall@K metrics

- **Main libraries**:
  - `pandas` – dataset loading and manipulation
  - `numpy` – numerical operations and averaging metrics
  - `tqdm` – progress tracking
  - `dotenv` – loading environment variables
  - Custom modules:
    - `TextProcessor`
    - `GeminiEmbedder`
    - `GoogleGemini`
    - `get_recall_K` (evaluation engine)

- **Configuration & Initialization**:
  - Loads environment variables:
    - `GEMINI_API_KEY`
    - `QDRANT_API_KEY`
    - `QDRANT_HOST_URL`
  - Initializes:
    - `TextProcessor` for query normalization
    - `GeminiEmbedder` for vector embedding and similarity search
    - `GoogleGemini` for LLM-based query refinement
  - Loads datasets:
    - Training labeled queries: `train-labeled-SHL.xlsx`
    - Test queries: `test-SHL.xlsx`
    - Assessment catalog: `assessments_details.csv`

- **Core function**:
  - `get_recommandation_links(query: str) -> dict`
    - Uses Gemini LLM to rewrite the query into a concise job description.
    - Preprocesses the refined text using `TextProcessor`.
    - Embeds the query and performs semantic search using Qdrant.
    - Retrieves top-K matching assessments via `dataset_id` mapping.
    - Returns:
      ```json
      {
        "query": "<original query>",
        "recommended_assessments": ["<assessment_url_1>", ...]
      }
      ```

- **Evaluation workflow**:
  1. **Prediction phase (optional / commented)**:
     - Generates recommendations for:
       - Training queries
       - Test queries
     - Saves results to:
       - `data/eval_results/trained_queries_recommendations.csv`
       - `data/eval_results/test_queries_recommendations.csv`

  2. **Recall@K computation**:
     - For each query:
       - Compares recommended assessment URLs with ground-truth URLs.
       - Computes Recall@K using:
         ```
         Recall@K = (# relevant assessments in top-K) / (total relevant assessments)
         ```
     - Uses `get_recall_K()` helper function.

  3. **Mean Recall@K**:
     - Aggregates Recall@K across all queries.
     - Ignores invalid or NaN values safely.
     - Prints final Mean Recall@K score.

- **Outputs**:
  - Console logs showing:
    - Recall@K per query
    - Final Mean Recall@K
  - CSV files containing recommendation predictions (when enabled).

- **Notes / caveats**:
  - LLM-based query refinement introduces slight non-determinism.
  - Rate-limiting (`time.sleep(1)`) is used to avoid API throttling.
  - Assumes assessment URLs in training data may include `/solutions`, which are normalized before comparison.
  - Script reads and writes evaluation CSVs in-place—ensure paths exist before execution.

![](/assets/evaluation_run.png)

### `run_pre_processing.py`

- **Purpose**:  
  Executes the preprocessing and vectorization pipeline for the SHL assessment catalog.  
  This script prepares assessment descriptions, generates embeddings, and stores them in a vector database (Qdrant) for downstream semantic search and recommendation.

- **Main libraries**:
  - `pandas` – loading and handling assessment metadata
  - `dotenv` – environment variable management
  - Custom modules:
    - `TextProcessor` – text cleaning, normalization, and lemmatization
    - `GeminiEmbedder` – embedding generation and vector storage/search

- **Configuration & Initialization**:
  - Loads environment variables:
    - `GEMINI_API_KEY`
    - `QDRANT_API_KEY`
    - `QDRANT_HOST_URL`
  - Initializes:
    - `TextProcessor` for preprocessing assessment descriptions
    - `GeminiEmbedder` configured with:
      - Gemini embedding model (`models/text-embedding-004`)
      - Qdrant Cloud collection (`collection_1`)

- **Data ingestion**:
  - Reads assessment catalog from:
    - `data/assessments_details.csv`
  - Required columns:
    - `desc` – assessment description text
    - `dataset_id` – unique identifier used for mapping vectors back to dataset rows

- **Preprocessing pipeline**:
  - Extracts all assessment descriptions.
  - Applies `process_text_for_embedding()`:
    - Cleans text
    - Tokenizes
    - Removes stopwords
    - Lemmatizes
  - Joins tokens back into normalized strings suitable for embedding models.

- **Embedding & Vector storage**:
  - Generates embeddings for all processed descriptions using Gemini embeddings.
  - Pushes vectors to Qdrant in batch mode.
  - Uses existing `dataset_id` values as vector IDs to maintain a stable mapping between:
    - Dataset rows
    - Stored vectors
    - Recommendation results

- **Sanity check / test query**:
  - Defines a sample hiring query:
    - Example: Senior Data Analyst with SQL, Excel, Python experience.
  - Preprocesses and embeds the query.
  - Performs similarity search against the embedded catalog.
  - Prints:
    - Similarity score
    - Matched assessment description (truncated)
    - Corresponding `dataset_id`

- **Outputs**:
  - Console logs showing:
    - Total number of assessments processed
    - Embedding and storage progress
    - Top-K similarity results for the test query
  - Vector data stored persistently in Qdrant Cloud.

- **Notes / caveats**:
  - This script recreates or overwrites the target Qdrant collection during embedding.
  - Ensure `dataset_id` values are already assigned (via `assign_ids.py`) before running.
  - Intended to be run once per catalog update, not per user query.

![](/assets/pre-processing-vectorizing-run.png)

### `run_recommand.py`

- **Purpose**:  
  Executes the online recommendation flow for the SHL Assessment Recommendation System.  
  Given a natural-language hiring query, this script retrieves the most relevant assessments using semantic similarity search over pre-embedded assessment descriptions stored in Qdrant.

- **Main libraries**:
  - `pandas` – loading assessment metadata
  - `dotenv` – environment variable management
  - Custom modules:
    - `TextProcessor` – preprocessing and normalization of query text
    - `GeminiEmbedder` – query embedding and vector similarity search

- **Configuration & Initialization**:
  - Loads environment variables:
    - `GEMINI_API_KEY`
    - `QDRANT_API_KEY`
    - `QDRANT_HOST_URL`
  - Initializes:
    - `TextProcessor` for query preprocessing
    - `GeminiEmbedder` configured with:
      - Gemini embedding model (`models/text-embedding-004`)
      - Existing Qdrant Cloud collection (`collection_1`)

- **Data loading**:
  - Reads assessment metadata from:
    - `data/assessments_details.csv`
  - Required columns:
    - `desc` – assessment description
    - `dataset_id` – unique identifier used to map vectors back to dataset rows

- **Query processing pipeline**:
  - Defines a natural-language hiring query (example provided in script).
  - Applies text preprocessing:
    - Cleaning
    - Tokenization
    - Stopword removal
    - Lemmatization
  - Converts processed query into an embedding using Gemini embeddings.

- **Recommendation & similarity search**:
  - Performs a Top-K similarity search (default `top_k=15`) against Qdrant.
  - Retrieves:
    - Similarity scores
    - Matched assessment descriptions
    - Corresponding `dataset_id` values

- **Output**:
  - Prints ranked recommendations to the console, including:
    - Cosine similarity score
    - Truncated assessment description
    - Dataset ID (used to fetch full metadata or URLs)

- **Design notes**:
  - This script assumes embeddings are already created and stored via `run_pre_processing.py`.
  - No vectors are written or modified—this is a read-only inference pipeline.
  - Intended to be used as:
    - A CLI test script
    - The backend logic for an API endpoint or Streamlit UI

- **Extensibility**:
  - Can be easily wrapped inside a REST API (FastAPI/Flask).
  - Can be directly called from Streamlit for interactive recommendations.

![](/assets/scrapper_run.png)

### `run_scrapper.py`

- **Purpose**:  
  Implements the data ingestion pipeline for the SHL Assessment Recommendation System.  
  This script crawls the SHL Product Catalog, extracts all *individual assessment* links, scrapes their detailed metadata, and stores the structured dataset locally for downstream processing.

- **Target source**:
  - SHL Product Catalog  
    https://www.shl.com/products/product-catalog/
  - Filter applied:
    - `type=1` → Individual Test Solutions
    - Excludes pre-packaged job solutions

- **Main libraries**:
  - `pandas` – dataset creation and persistence
  - `tqdm` – progress monitoring
  - `concurrent.futures.ThreadPoolExecutor` – parallel scraping
  - Custom module:
    - `AssessmentScrapperEngine` – HTML fetching and parsing logic

- **Scraping strategy**:
  - Pagination logic:
    - Each catalog page contains **12 assessments**
    - Pages are accessed using `start={page_index}` with step size `12`
    - Total pages crawled: `32`
  - Listing URLs generated as:
    ```
    https://www.shl.com/products/product-catalog/?start={i}&type=1
    ```

- **Pipeline stages**:

  **1. Assessment listing extraction**
  - Fetches catalog listing pages in parallel (`max_workers=16`)
  - Extracts individual assessment detail page URLs
  - Aggregates all assessment links into a single list

  **2. Assessment detail extraction**
  - Each assessment page is fetched in parallel (`max_workers=6`)
  - Extracted fields:
    - `link`
    - `heading`
    - `desc`
    - `job_levels`
    - `languages`
    - `assessment_length`
    - `test_type`
    - `remote_testing`
  - Errors are caught and logged without stopping execution

- **Data output**:
  - Saves the final structured dataset as:
    ```
    data/assessments_details.csv
    ```
  - Each row represents one assessment with full metadata

- **Concurrency considerations**:
  - Uses conservative thread pools to avoid rate limiting
  - Two-stage parallelism:
    - Listing pages (fast, lightweight)
    - Detail pages (slower, content-heavy)

- **Design notes**:
  - This script is intended to be run **once or periodically**
  - Output CSV acts as the single source of truth for:
    - ID assignment
    - Text preprocessing
    - Embedding generation
  - Dataset correctness is critical for downstream recommendation quality

- **Typical usage**:
  ```bash
  python run_scrapper.py
```

![](/assets/scrapper_run.png)