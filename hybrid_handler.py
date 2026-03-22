from chroma_connect import user_question
from duckdb_connect import sql_answer
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

def combined_results(question):
    prompt = build_hybrid_split_prompt(question)
    result = call_ollama_json(OLLAMA_MODEL, OLLAMA_GENERATE_URL, prompt)

    if "structured_question" not in result or "semantic_question" not in result:
        raise ValueError(f"LLM returned unexpected JSON keys: {list(result.keys())}")

    structured_res = sql_answer(result["structured_question"])
    product_ids = None
    product_names = None
    brand_names = None
    categories = None
    if not structured_res.empty:
        product_ids = list(structured_res["product_id"])
        product_names = list(structured_res["product_name"])
        brand_names = list(structured_res["brand_name"])
        categories = list(structured_res["primary_category"])

    docs, _ = user_question(result["semantic_question"], product_ids)
    payload = {
        "route": "hybrid",
        "question": question,
        "structured_question": result["structured_question"],
        "semantic_question": result["semantic_question"],
        "retrieval_scope": {
            "filter_type": "product_ids",
            "values": product_ids
        },
        "structured": structured_res,
        "semantic": docs,
        "combined_context": {
            "top_rows": structured_res.head(3).to_dict("records") if not structured_res.empty else None,
            "review_snippets": docs[:5] if docs else None,
            "entities": {
                "product_ids": product_ids,
                "product_names": product_names,
                "brand_names": brand_names,
                "categories": categories
            }
        }
    }
    return payload
