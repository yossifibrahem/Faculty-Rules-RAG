from typing import Dict, List, Tuple, Optional
from RAG.embedding import VectorDB, EmbeddingClient

class DatabaseManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.databases = {}  # Initialize empty dict
            cls._instance.__annotations__ = {'databases': Dict[str, VectorDB]}  # Add type annotation
        return cls._instance

    def create_database(self, name: str, embedding_client: Optional[EmbeddingClient] = None) -> VectorDB:
        if name not in self.databases:
            self.databases[name] = VectorDB(name, embedding_client)
        return self.databases[name]

    def get_database(self, name: str) -> Optional[VectorDB]:
        return self.databases.get(name)

    def search_database(self, name: str, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        db = self.get_database(name)
        if db is None:
            return []
        return db.search(query, top_k)

    @classmethod
    def get_instance(cls) -> 'DatabaseManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
