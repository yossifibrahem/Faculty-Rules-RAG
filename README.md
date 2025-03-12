designed for semantic search across different types of content (rules, FAQs) using vector embeddings for more accurate matching than traditional keyword search.
# 1. ``SearchRules.py``
**Main Functions:**

- ``initialize_rules_db``: Sets up or loads a vector database for rules.
- ``search_rules``: Searches the database with a query.
- ``load_rules``: Loads and splits rules from a text file

**The file demonstrates usage by:**
1. Creating databases for both rules and FAQs
2. Performing searches across both databases
3. Comparing search results

# 2. ``embedding.py``
**Contains three main classes:**

1. ``EmbeddingClient``
- Handles text-to-vector embedding using OpenAI's API
- Features caching for performance
```python
@lru_cache(maxsize=1000)
def get_embedding(self, text: str) -> List[float]
```

2. ``TextSplitter``
- Static utility for splitting text into segments
- Uses ``<split>`` markers to separate content

3. ``VectorDB``
-Core vector database implementation
-Key features:
    - Text storage and vector representation
    - Similarity search using cosine similarity
    - Save/load functionality for persistence

# 3. ``database_manager.py``
Implements a singleton pattern for managing multiple vector databases:

- Uses the singleton pattern via _instance
- Manages multiple VectorDB instances
- Provides methods for:
    - ``create_database()``
    - ``get_database()``
    - ``search_database()``


**Key Features**
1. Vector Similarity Search
    - Converts text to vectors using embeddings
    - Uses cosine similarity for matching
    - Caches results for performance

2. Database Management
    - Multiple database support
    - Persistent storage
    - Singleton pattern for global access