from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from uuid import uuid4
import numpy as np
from httpx import Timeout
from .logger_config import logger
import pandas as pd

class QdrantClientWrapper:
    """
    A lightweight wrapper around QdrantClient for storing and searching vector embeddings.
    """

    def __init__(self, url: str = None, api_key: str = None):
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=3000
        )
        logger.info(f"Connected to Qdrant at {url}")

    # COLLECTION MANAGEMENT
    def create_collection(self, collection_name: str, vector_size: int):
        existing = [c.name for c in self.client.get_collections().collections]

        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created new collection: {collection_name}")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

    def recreate_collection(self, collection_name: str, vector_size: int):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.warning(f"Recreated collection: {collection_name}")

    # VECTOR INSERTION (BATCHED)
    def add_vector_embeddings(
        self,
        collection_name: str,
        embeddings: list,
        payloads: list = None,
        batch_size: int = 64,
    ):
        if payloads is None:
            payloads = [{} for _ in embeddings]

        embeddings = np.asarray(embeddings, dtype=np.float32).tolist()
        points = []

        for embedding, payload in zip(embeddings, payloads):
            points.append(
                PointStruct(
                    id = payload.get("dataset_id", str(uuid4())),
                    vector=embedding,
                    payload=payload,
                )
            )

        total = len(points)
        logger.info(f"Upserting {total} vectors into '{collection_name}'")

        for i in range(0, total, batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            logger.info(
                f"Inserted batch {i // batch_size + 1} "
                f"({len(batch)} vectors)"
            )

        logger.info(f"Successfully added {total} vectors to '{collection_name}'")

    def search_vectors(
        self,
        collection_name: str,
        query_vector: list,
        top_k: int = 5,
    ):
        query_vector = np.asarray(query_vector, dtype=np.float32).tolist()

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        )

        logger.info(
            f"Retrieved top {top_k} similar vectors "
            f"from collection '{collection_name}', "
            # f"results: {response}"
        )

        return [
            {
                "id": p.id,
                "score": float(p.score),
                "payload": p.payload,
            }
            for p in response.points   
        ]

    # UTILITIES
    def count_points(self, collection_name: str) -> int:
        return self.client.count(
            collection_name=collection_name,
            exact=True,
        ).count

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name=collection_name)
        logger.warning(f"Deleted collection: {collection_name}")