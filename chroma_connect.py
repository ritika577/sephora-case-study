import chromadb
from ollama_utils import embed
from config import EMBED_MODEL, OLLAMA_EMBED_URL, CHROMA_PATH, CHROMA_COLLECTION

client = chromadb.PersistentClient(path=CHROMA_PATH)
coll = client.get_or_create_collection(CHROMA_COLLECTION)


def user_question(question, product_ids=None, limit=5):
    try:
        question = question.strip() if question else None
        if not question:
            return None, "question is not valid"

        query_args = {
            "query_embeddings": [embed(EMBED_MODEL, OLLAMA_EMBED_URL, question)],
            "n_results": limit,
        }

        # If product_id is stored in metadata
        if product_ids is not None:
            query_args["where"] = {
                "product_id": {"$in": list(product_ids)}
            }
        results = coll.query(**query_args)
        docs = results["documents"]
        return docs, None
    except Exception as e:
        print(f"Error while fetching the required data for user's input from chroma : {str(e)}")
        return None, str(e)


    
