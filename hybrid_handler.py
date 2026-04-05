import pandas as pd
from chroma_connect import user_question
from duckdb_connect import process_structured_question
from ollama_utils import call_ollama_json
from config import OLLAMA_MODEL, OLLAMA_GENERATE_URL
        

def build_hybrid_split_prompt(question: str) -> str:
    return f"""
You are a question decomposition assistant for a Sephora analytics system.

A hybrid question contains:
1. a structured analytics part that can be answered using SQL
2. a semantic part that requires review text understanding

Split the user's hybrid question into:
- structured_question
- semantic_question

Rules:
- Return valid JSON only
- Keep both questions short and precise
- If the semantic question depends on the structured result, write it in a general form

Example:
User: Which product has the most loves and what do users say about it?
Output:
{{
  "structured_question": "Which product has the most loves?",
  "semantic_question": "What do users say about this product?"
}}

User question:
{question}
""".strip()

def combined_results(question: str) -> dict:
    prompt = build_hybrid_split_prompt(question)
    try:
        result = call_ollama_json(OLLAMA_MODEL, OLLAMA_GENERATE_URL, prompt)
        if "structured_question" not in result or "semantic_question" not in result:
            raise ValueError(f"LLM returned unexpected JSON keys: {list(result.keys())}")

        structured_res, structured_err, structured_sql = process_structured_question(result["structured_question"])
        product_ids = None
        semantic_q = result["semantic_question"]

        if structured_res is not None and not structured_res.empty:
            if "product_id" in structured_res.columns:
                product_ids = list(structured_res["product_id"])
            # Enrich vague semantic question with actual names from structured results
            name_cols = [c for c in ["product_name", "brand_name"] if c in structured_res.columns]
            if name_cols:
                names = structured_res[name_cols].head(5).apply(
                    lambda row: " ".join(str(v) for v in row if pd.notna(v)), axis=1
                ).tolist()
                context = ", ".join(names)
                semantic_q = f"{semantic_q} (specifically about: {context})"

        semantic_res, semantic_err = user_question(semantic_q, product_ids)

        payload = {
        "structured": structured_res,
        "semantic": semantic_res,
        "sql": structured_sql,
        "error": {
                "split": None,
                "structured": structured_err,
                "semantic": semantic_err
            }
        }
        return payload
    except Exception as e:
        return {
        "structured": None,
        "semantic": None ,
        "error": {
                "split": str(e),
                "structured": None,
                "semantic": None
            }       
        }
