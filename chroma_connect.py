import chromadb
from ollama_utils import embed
from config import EMBED_MODEL, OLLAMA_EMBED_URL, CHROMA_PATH, CHROMA_COLLECTION

# Lazy connection — created on first use, not at import time
_client = None
_coll = None


def _get_collection():
    global _client, _coll
    if _coll is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _coll = _client.get_or_create_collection(CHROMA_COLLECTION)
    return _coll


def user_question(question: str, product_ids: list | None = None, limit: int = 5) -> tuple:
    try:
        question = question.strip() if question else None
        if not question:
            return None, "question is not valid"

        query_args = {
            "query_embeddings": [embed(EMBED_MODEL, OLLAMA_EMBED_URL, question)],
            "n_results": limit,
        }

        if product_ids is not None:
            query_args["where"] = {
                "product_id": {"$in": list(product_ids)}
            }
        results = _get_collection().query(**query_args)
        docs = results["documents"]
        return docs, None
    except Exception as e:
        print(f"Error while fetching the required data for user's input from chroma: {str(e)}")
        return None, str(e)

