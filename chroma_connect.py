import pandas as pd
import chromadb
from ollama_utils import embed

df = pd.read_csv("analysis_output/clean_merged.csv").fillna("")

client = chromadb.PersistentClient(path="chroma_store")
coll = client.get_or_create_collection("sephora_merged")
limit = 2

OLLAMA_EMBED_URL ="http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


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

    batch = 500
    for i in range(137500, len(df), batch):
        print("chunk: start: ",i,"end: ",i+batch)
        chunk_ids = ids[i:i+batch]
        chunk_docs = documents[i:i+batch]
        chunk_texts = texts[i:i+batch]
        chunk_metas = metas[i:i+batch]
        filtered = [(i, d, t, m) for i, d, t, m in zip(chunk_ids, chunk_docs, chunk_texts, chunk_metas) if t.strip()]
        chunk_ids   = [x[0] for x in filtered]
        chunk_docs  = [x[1] for x in filtered]
        chunk_texts = [x[2] for x in filtered]
        chunk_metas = [x[3] for x in filtered]
        chunk_embs = [embed(EMBED_MODEL,OLLAMA_EMBED_URL,t) for t in chunk_texts]
        print("chunk_embs: ",chunk_embs)
        coll.add(ids=chunk_ids, documents=chunk_docs, embeddings=chunk_embs, metadatas=chunk_metas)


initialization()

def user_question(question, product_ids=None, limit=5):
    question = question.strip() if question else None
    if not question:
        return None, "question is not valid"

    query_args = {
        "query_embeddings": [embed(question)],
        "n_results": limit,
    }

    # If product_id is stored in metadata
    if product_ids:
        query_args["where"] = {
            "product_id": {"$in": product_ids}
        }
    results = coll.query(**query_args)
    docs = results["documents"]
    return docs, None
