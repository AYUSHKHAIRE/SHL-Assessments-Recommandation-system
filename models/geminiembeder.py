from typing import List, Optional
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

from .logger_config import logger
from .vector import QdrantClientWrapper


class GeminiEmbedder:
    def __init__(
        self,
        collection_name: str = "collection_1",
        model_name: str = "models/text-embedding-004",
        gemini_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")

        genai.configure(api_key=gemini_api_key)

        self.model_name = model_name
        self.collection_name = collection_name

        self.all_embeddings: List[np.ndarray] = []
        self.chunks: List[str] = []

        self.qdrant_enabled = True
        self.qdrant = None

        # Try Qdrant
        try:
            if qdrant_url:
                self.qdrant = QdrantClientWrapper(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                )
                self.qdrant_enabled = True
                logger.info(f"Qdrant connected. Using collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Qdrant not available ({e}). Falling back to in-memory search.")

        logger.info(f"GeminiEmbedder initialized with model={self.model_name}")

    def _embed_single_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed batch of texts using Gemini.
        """
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embeddings.append(result["embedding"])

        return np.array(embeddings, dtype=np.float32)

    def embed_chunks(self, chunks: List[str], ids_list: List[str] = None, batch_size: int = 16):
        """
        Embed and optionally store chunks.
        """
        if not chunks:
            logger.warning("embed_chunks called with empty chunks.")
            return []

        self.chunks = list(chunks)
        self.all_embeddings = []

        chunk_batches = [
            self.chunks[i : i + batch_size]
            for i in range(0, len(self.chunks), batch_size)
        ]
        id_batches = [
            ids_list[i : i + batch_size]
            for i in range(0, len(ids_list), batch_size)
        ]

        for batch in tqdm(zip(chunk_batches, id_batches), desc="Embedding chunks using Gemini"):
            batch_chunks, batch_ids = batch
            batch_embeddings = self._embed_single_batch(batch_chunks)
            self.all_embeddings.extend(batch_embeddings)

        if not self.all_embeddings:
            logger.warning("No embeddings were created.")
            return []

        # Store in Qdrant
        if self.qdrant_enabled:
            vector_size = self.all_embeddings[0].shape[0]

            self.qdrant.recreate_collection(
                self.collection_name,
                vector_size,
            )

            payloads = [
                {
                    "text": chunk,
                    "dataset_id": dataset_id
                } for chunk, dataset_id in zip(self.chunks, ids_list)]
            vectors = [vec.tolist() for vec in self.all_embeddings]

            self.qdrant.add_vector_embeddings(
                self.collection_name,
                vectors,
                payloads,
            )

            logger.info(f"Stored {len(vectors)} vectors in Qdrant.")
        else:
            logger.info(f"Stored {len(self.all_embeddings)} vectors in memory.")

        return self.all_embeddings

    # Query helpers
    def get_user_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for user query.
        """
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="RETRIEVAL_QUERY",
        )
        return np.array(result["embedding"], dtype=np.float32)

    def get_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search via Qdrant or fallback to in-memory cosine similarity.
        """
        if self.qdrant_enabled:
            logger.info("Searching via Qdrant...")
            results = self.qdrant.search_vectors(
                self.collection_name,
                query_embedding.tolist(),
                top_k=top_k,
            )
            scores = [r["score"] for r in results]
            texts = [r["payload"]["text"] for r in results]
            ids = [r["payload"]["dataset_id"] for r in results]
            return scores, texts, ids

        # In-memory fallback
        if not self.all_embeddings:
            raise ValueError("No embeddings available. Run embed_chunks() first.")

        sims = cosine_similarity(
            [query_embedding],
            np.array(self.all_embeddings),
        )[0]

        idxs = np.argsort(sims)[-top_k:][::-1]
        scores = [float(sims[i]) for i in idxs]
        texts = [self.chunks[i] for i in idxs]

        return scores, texts
