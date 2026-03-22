import pandas as pd
import chromadb
from ollama_utils import embed
from config import EMBED_MODEL, OLLAMA_EMBED_URL, CHROMA_PATH, CHROMA_COLLECTION, BATCH_SIZE

df = pd.read_csv("analysis_output/clean_merged.csv").fillna("")

client = chromadb.PersistentClient(path=CHROMA_PATH)
coll = client.get_or_create_collection(CHROMA_COLLECTION)


def initialization()-> None:
    # choose the text to embed (for semantic search)
    texts = (df["review_text_clean"] + " " +df["product_name"]+ " " +df["brand_name"]+ " " +df["ingredients"]\
    + " " +df["highlights"]+ " " +df["primary_category"]+ " " +df["secondary_category"]+ " " +df["tertiary_category"]\
    + " " +df["size_unit"]+ " " +df["rating_bucket"]+ " " +df["hair_color"]\
    + " " +df["skin_type"]+ " " +df["eye_color"]+ " " +df["skin_tone"]).astype(str).tolist()
    ids = df.index.astype(str).tolist()

    # store the COMPLETE row as the document (JSON string)
    documents = df.apply(lambda row: row.to_json(), axis=1).tolist()

    # (optional) store a few filterable fields as metadata
    metas = df[["product_id","brand_name","primary_category"]].astype(str).to_dict("records")

    for i in range(0, len(df), BATCH_SIZE):
        print("chunk: start: ",i,"end: ",i+batch)
        chunk_ids = ids[i:i+BATCH_SIZE]
        chunk_docs = documents[i:i+BATCH_SIZE]
        chunk_texts = texts[i:i+BATCH_SIZE]
        chunk_metas = metas[i:i+BATCH_SIZE]
        filtered = [(i, d, t, m) for i, d, t, m in zip(chunk_ids, chunk_docs, chunk_texts, chunk_metas) if t.strip()]
        chunk_ids   = [x[0] for x in filtered]
        chunk_docs  = [x[1] for x in filtered]
        chunk_texts = [x[2] for x in filtered]
        chunk_metas = [x[3] for x in filtered]
        chunk_embs = [embed(EMBED_MODEL,OLLAMA_EMBED_URL,t) for t in chunk_texts]
        print("chunk_embs: ",chunk_embs)
        coll.add(ids=chunk_ids, documents=chunk_docs, embeddings=chunk_embs, metadatas=chunk_metas)


if __name__ == "__main__":
    initialization()

def user_question(question, product_ids=None, limit=5):
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
