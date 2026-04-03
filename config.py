import os

# project root = directory where config.py lives
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# OLLAMA
# =========================================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = OLLAMA_BASE_URL + "/api/generate"
OLLAMA_EMBED_URL = OLLAMA_BASE_URL + "/api/embed"
OLLAMA_TAGS_URL = OLLAMA_BASE_URL + "/api/tags"
OLLAMA_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text:latest"

# =========================================================
# DUCKDB
# =========================================================
DB_PATH = os.path.join(PROJECT_ROOT, "sephora.duckdb")
CSV_PATH = os.path.join(PROJECT_ROOT, "analysis_output", "clean_merged.csv")
TABLE_NAME = "sephora"
DEFAULT_LIMIT = 50
MAX_SQL_RETRIES = 1

# =========================================================
# CHROMADB
# =========================================================
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_store")
CHROMA_COLLECTION = "sephora_merged"
BATCH_SIZE = 500
MAX_EMBED_CHARS = 8000
