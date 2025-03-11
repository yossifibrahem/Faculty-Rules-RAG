from embedding import TextSplitter
from database_manager import DatabaseManager
import os
from typing import Dict, List, Tuple, Optional
from embedding import VectorDB

# Constants
RULES_DB_PATH = 'rules_db.pkl'
RULES_FILE_PATH = 'rules.txt'
DEFAULT_DB_NAME = 'rules'

def initialize_rules_db(rules_file_path: str, db_path: str, db_name: str = DEFAULT_DB_NAME) -> None:
    """Initialize or load the rules database."""
    manager = DatabaseManager.get_instance()
    
    try:
        if os.path.exists(db_path):
            db = VectorDB.load(db_path, name=db_name)
            manager.databases[db_name] = db
        else:
            db = manager.create_database(db_name)
            rules = load_rules(rules_file_path)
            db.add_texts(rules)
            db.save(db_path)
    except Exception as e:
        print(f"Warning: Failed to initialize rules database: {str(e)}")

def search_rules(query: str, db_name: str = DEFAULT_DB_NAME, top_k: int = 3) -> List[Dict[str, any]]:
    """
    Search rules database with the given query.
    Returns list of (rule_text, similarity_score) tuples.
    """
    manager = DatabaseManager.get_instance()
    try:
        return manager.search_database(db_name, query, top_k)
    except Exception as e:
        print(f"Warning: Rules search failed: {str(e)}")
        return []

def load_rules(file_path: str) -> List[str]:
    """Load text from the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return TextSplitter.split(content)
    except Exception as e:
        print(f"Warning: Failed to load data: {str(e)}")
        return []

# Example usage
manager = DatabaseManager.get_instance()

# Create and initialize different databases
initialize_rules_db('text\\rules.txt', 'vector_db\\rules_db.pkl', 'rules')
initialize_rules_db('text\\FAQ.txt', 'vector_db\\FAQ_db.pkl', 'FAQ')