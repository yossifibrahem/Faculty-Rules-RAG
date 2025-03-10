from typing import List, Tuple, Dict, Optional
from openai import OpenAI
import numpy as np
from functools import lru_cache

class EmbeddingClient:
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", api_key: str = "dummy-key"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._cache: Dict[str, List[float]] = {}

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str, model: str = "text-embedding-nomic-embed-text-v1.5") -> List[float]:
        """Get embedding for text with caching for better performance."""
        text = self._normalize_text(text)
        if text in self._cache:
            return self._cache[text]
        
        try:
            embedding = self.client.embeddings.create(input=[text], model=model).data[0].embedding
            self._cache[text] = embedding
            return embedding
        except Exception as e:
            raise Exception(f"Failed to get embedding: {str(e)}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for consistent embedding."""
        return text.strip().replace("\n", " ")

class TextSplitter:
    @staticmethod
    def split(text: str, split_marker: str = "<split>") -> List[str]:
        """Split text into parts using a marker."""
        parts = []
        current_part = []

        for line in text.splitlines():
            if split_marker in line:
                if current_part:
                    parts.append("\n".join(current_part).strip())
                current_part = []
            else:
                current_part.append(line)

        if current_part:
            parts.append("\n".join(current_part).strip())

        return [part for part in parts if part]

class VectorDB:
    _instance = None
    
    def __new__(cls, embedding_client: Optional[EmbeddingClient] = None):
        if cls._instance is None:
            cls._instance = super(VectorDB, cls).__new__(cls)
            cls._instance.embedding_client = embedding_client or EmbeddingClient()
            cls._instance.vectors = []
            cls._instance.texts = []
        return cls._instance

    def __init__(self, embedding_client: Optional[EmbeddingClient] = None):
        # Skip initialization if already initialized
        pass

    def add_texts(self, texts: List[str]) -> None:
        """Add multiple texts to the database in batch."""
        for text in texts:
            embedding = self.embedding_client.get_embedding(text)
            self.vectors.append(np.array(embedding))
            self.texts.append(text)

    def add_text(self, text: str) -> None:
        """Add a single text to the database."""
        self.add_texts([text])

    def search(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """Search for similar texts using cosine similarity."""
        if not self.vectors:
            return []

        query_embedding = np.array(self.embedding_client.get_embedding(query))
        similarities = [
            {
                "content":text, "similarity":self._cosine_similarity(query_embedding, vec)
            }
            for text, vec in zip(self.texts, self.vectors)
        ]
        
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def clear(self) -> None:
        """Clear the database."""
        self.vectors.clear()
        self.texts.clear()

    def save(self, file_path: str) -> None:
        """Save the database to a file."""
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump((self.vectors, self.texts), f)

    @classmethod
    def load(cls, file_path: str, embedding_client: Optional[EmbeddingClient] = None) -> 'VectorDB':
        """Load a database from a file."""
        import pickle
        db = cls(embedding_client)
        with open(file_path, 'rb') as f:
            db.vectors, db.texts = pickle.load(f)
        return db

    @classmethod
    def get_instance(cls) -> 'VectorDB':
        """Get the singleton instance of VectorDB."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance




