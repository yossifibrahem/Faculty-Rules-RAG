from embedding import TextSplitter, VectorDB
import os
from typing import List, Tuple, Optional

# Constants
RULES_DB_PATH = os.path.join(os.path.dirname(__file__), 'rules_db.pkl')
RULES_FILE_PATH = os.path.join(os.path.dirname(__file__), 'rules.txt')

def initialize_rules_db() -> None:
    """Initialize or load the rules database."""
    db = VectorDB.get_instance()
    
    try:
        if os.path.exists(RULES_DB_PATH):
            VectorDB.load(RULES_DB_PATH)
        else:
            if not os.path.exists(RULES_FILE_PATH):
                raise FileNotFoundError(f"Rules file not found: {RULES_FILE_PATH}")
            
            rules = load_rules(RULES_FILE_PATH)
            db.add_texts(rules)
            db.save(RULES_DB_PATH)
    except Exception as e:
        print(f"Warning: Failed to initialize rules database: {str(e)}")

def search_rules(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Search rules database with the given query.
    Returns list of (rule_text, similarity_score) tuples.
    """
    db = VectorDB.get_instance()
    try:
        return db.search(query, top_k)
    except Exception as e:
        print(f"Warning: Rules search failed: {str(e)}")
        return []

def load_rules(file_path: str) -> List[str]:
    """Load rules from the rules file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return TextSplitter.split(content)
    except Exception as e:
        print(f"Warning: Failed to load rules: {str(e)}")
        return []

# Initialize database when module is imported
initialize_rules_db()
