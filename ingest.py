"""
Ingestion pipeline: reads raw CSVs, cleans data, loads into DuckDB and ChromaDB.

Usage:
    python ingest.py                        # full pipeline (clean + duckdb + chroma)
    python ingest.py --clean                # merge + clean only
    python ingest.py --duckdb               # reload DuckDB from existing clean CSV
    python ingest.py --chroma               # reload ChromaDB from existing clean CSV
    python ingest.py --clean --duckdb       # clean then reload DuckDB
    python ingest.py --clean --chroma       # clean then reload ChromaDB
    python ingest.py --duckdb --chroma      # reload both DBs from existing CSV
    python ingest.py --clean --duckdb --chroma  # clean + reload both (same as full)

Flags can be combined freely. Steps run in order: clean -> duckdb -> chroma.
"""

import os
import sys
import glob
import pandas as pd
import duckdb
import chromadb
from data_cleaning import clean_text
from ollama_utils import embed, startup_checks
from config import (
    DB_PATH, CSV_PATH, TABLE_NAME,
    EMBED_MODEL, OLLAMA_EMBED_URL,
    CHROMA_PATH, CHROMA_COLLECTION, BATCH_SIZE,
    DATA_DIR, ANALYSIS_OUTPUT, PRODUCTS_FILE, REVIEWS_PATTERN,
)


# =========================================================
# STEP 1: MERGE RAW CSVs
# =========================================================
def merge_raw_csvs() -> pd.DataFrame:
    """Read product_info.csv and all reviews_*.csv, merge on product_id."""
    print("[ingest] Reading raw CSVs...")
    products = pd.read_csv(PRODUCTS_FILE)

    review_files = sorted(glob.glob(REVIEWS_PATTERN))
    if not review_files:
        raise FileNotFoundError(f"No review files found matching {REVIEWS_PATTERN}")

    reviews = pd.concat([pd.read_csv(f) for f in review_files], ignore_index=True)
    print(f"[ingest] Products: {len(products)} rows, Reviews: {len(reviews)} rows")

    # left join: keep all products, attach matching reviews
    df = products.merge(reviews, on="product_id", how="left")

    # drop duplicate columns from merge (e.g. brand_name_y) and rename _x columns
    cols_to_drop = [c for c in df.columns if c.endswith("_y")]
    df = df.drop(columns=cols_to_drop)
    df = df.rename(columns={c: c[:-2] for c in df.columns if c.endswith("_x")})

    # drop rows with no price (can't analyze them)
    df = df.dropna(subset=["price_usd"]).copy()
    print(f"[ingest] Merged: {len(df)} rows")
    return df


# =========================================================
# STEP 2: CLEAN AND SAVE TO CSV
# =========================================================
def clean_and_save(df: pd.DataFrame) -> pd.DataFrame:
    """Run data_cleaning.clean_text() and save result to CSV_PATH."""
    print("[ingest] Cleaning data...")
    os.makedirs(ANALYSIS_OUTPUT, exist_ok=True)
    clean_df = clean_text(df)
    clean_df.to_csv(CSV_PATH, index=False)
    print(f"[ingest] Saved cleaned data to {CSV_PATH} ({len(clean_df)} rows)")
    return clean_df


# =========================================================
# STEP 3: LOAD INTO DUCKDB
# =========================================================
def load_duckdb(csv_path: str = CSV_PATH) -> None:
    """Create/replace the DuckDB table from the cleaned CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cleaned CSV not found at {csv_path}. Run the full pipeline first."
        )

    print(f"[ingest] Loading {csv_path} into DuckDB ({DB_PATH})...")
    con = duckdb.connect(DB_PATH)
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT * FROM read_csv_auto('{csv_path}');
    """)
    row_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    con.close()
    print(f"[ingest] DuckDB table '{TABLE_NAME}' loaded with {row_count} rows")


# =========================================================
# STEP 4: LOAD INTO CHROMADB
# =========================================================
def load_chromadb(csv_path: str = CSV_PATH) -> None:
    """Embed all rows and load into ChromaDB collection."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cleaned CSV not found at {csv_path}. Run the full pipeline first."
        )

    # embedding requires Ollama to be running with the embed model
    startup_checks(EMBED_MODEL)

    print(f"[ingest] Reading {csv_path} for ChromaDB ingestion...")
    df = pd.read_csv(csv_path).fillna("")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # delete existing collection to start fresh (avoids duplicates)
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print(f"[ingest] Deleted existing collection '{CHROMA_COLLECTION}'")
    except ValueError:
        pass
    coll = client.create_collection(CHROMA_COLLECTION)

    # build embedding text by concatenating key text columns
    text_columns = [
        "review_text_clean", "product_name", "brand_name", "ingredients",
        "highlights", "primary_category", "secondary_category", "tertiary_category",
        "size_unit", "rating_bucket", "hair_color", "skin_type", "eye_color", "skin_tone",
    ]
    existing_cols = [c for c in text_columns if c in df.columns]
    texts = df[existing_cols].astype(str).agg(" ".join, axis=1).tolist()

    # row index as ID, full row JSON as document, key fields as filterable metadata
    ids = df.index.astype(str).tolist()
    documents = df.apply(lambda row: row.to_json(), axis=1).tolist()
    metas = df[["product_id", "brand_name", "primary_category"]].astype(str).to_dict("records")

    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[ingest] Embedding and loading {len(df)} rows in {total_batches} batches...")

    for batch_num, i in enumerate(range(0, len(df), BATCH_SIZE), 1):
        chunk_ids = ids[i:i + BATCH_SIZE]
        chunk_docs = documents[i:i + BATCH_SIZE]
        chunk_texts = texts[i:i + BATCH_SIZE]
        chunk_metas = metas[i:i + BATCH_SIZE]

        # filter out rows with empty embedding text
        filtered = [
            (cid, doc, txt, meta)
            for cid, doc, txt, meta in zip(chunk_ids, chunk_docs, chunk_texts, chunk_metas)
            if txt.strip()
        ]
        if not filtered:
            continue

        chunk_ids = [x[0] for x in filtered]
        chunk_docs = [x[1] for x in filtered]
        chunk_texts = [x[2] for x in filtered]
        chunk_metas = [x[3] for x in filtered]

        chunk_embs = [embed(EMBED_MODEL, OLLAMA_EMBED_URL, t) for t in chunk_texts]
        coll.add(
            ids=chunk_ids,
            documents=chunk_docs,
            embeddings=chunk_embs,
            metadatas=chunk_metas,
        )
        print(f"[ingest] Batch {batch_num}/{total_batches} done ({len(chunk_ids)} rows)")

    print(f"[ingest] ChromaDB collection '{CHROMA_COLLECTION}' loaded successfully")


# =========================================================
# FULL PIPELINE
# =========================================================
def run_full_pipeline() -> None:
    """Full pipeline: merge raw CSVs -> clean -> DuckDB -> ChromaDB."""
    merged = merge_raw_csvs()
    clean_and_save(merged)
    load_duckdb()
    load_chromadb()
    print("[ingest] Full pipeline complete!")


# =========================================================
# CLI ENTRY POINT
# =========================================================
if __name__ == "__main__":
    valid_flags = {"--clean", "--duckdb", "--chroma"}
    args = set(sys.argv[1:])
    unknown = args - valid_flags

    if unknown:
        print(f"Unknown arguments: {unknown}")
        print(__doc__)
        sys.exit(1)

    if not args:
        # No flags = full pipeline
        run_full_pipeline()
    else:
        # Run each requested step in order
        if "--clean" in args:
            merged = merge_raw_csvs()
            clean_and_save(merged)
        if "--duckdb" in args:
            load_duckdb()
        if "--chroma" in args:
            load_chromadb()

    print("[ingest] Done.")
