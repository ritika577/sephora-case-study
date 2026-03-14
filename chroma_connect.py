import pandas as pd
import chromadb
import requests

df = pd.read_csv("analysis_output/clean_merged.csv").fillna("")

client = chromadb.PersistentClient(path="chroma_store")
coll = client.get_or_create_collection("sephora_merged")
limit = 2

def embed(text, model="nomic-embed-text"):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": model, "input": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embeddings"][0]

def initialization()-> None:
    # choose the text to embed (for semantic search)
    texts = df["review_text_clean"] + " " +df["product_name"]+ " " +df["brand_name"]+ " " +df["ingredients"]\
    + " " +df["highlights"]+ " " +df["primary_category"]+ " " +df["secondary_category"]+ " " +df["tertiary_category"]\
    + " " +df["size_unit"]+ " " +df["rating_bucket"]+ " " +df["review_text_clean"]+ " " +df["hair_color"]\
    + " " +df["skin_type"]+ " " +df["eye_color"]+ " " +df["skin_tone"].astype(str).tolist()
    ids = df.index.astype(str).tolist()

    # store the COMPLETE row as the document (JSON string)
    documents = df.apply(lambda row: row.to_json(), axis=1).tolist()

    # (optional) store a few filterable fields as metadata
    metas = df[["product_id","brand_name","primary_category"]].astype(str).to_dict("records")

    batch = 500
    for i in range(0, len(df), batch):
        print("chunk: start: ",i,"end: ",i+batch)
        chunk_ids = ids[i:i+batch]
        chunk_docs = documents[i:i+batch]
        chunk_texts = texts[i:i+batch]
        chunk_metas = metas[i:i+batch]
        chunk_embs = [embed(t) for t in chunk_texts]
        print("chunk_embs: ",chunk_embs)
        coll.add(ids=chunk_ids, documents=chunk_docs, embeddings=chunk_embs, metadatas=chunk_metas)


# initialization()

def user_question(question):
    question = question.strip()
    if question:
        results = coll.query(
    query_embeddings=[embed(question)],
    n_results= limit,
    )
        docs = results["documents"]
        return docs , None
    
    return None , "question is not valid"

if __name__ == "__main__":
    result = user_question("i have a dry skin so suggest moisturizer for it?")
    print(result)