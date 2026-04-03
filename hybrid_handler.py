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

def combined_results(question):
    prompt = build_hybrid_split_prompt(question)
    try:
        result = call_ollama_json(OLLAMA_MODEL, OLLAMA_GENERATE_URL, prompt)
        if "structured_question" not in result or "semantic_question" not in result:
            raise ValueError(f"LLM returned unexpected JSON keys: {list(result.keys())}")

        structured_res,structured_err = process_structured_question(result["structured_question"])
        product_ids = None
        if structured_res is not None and not structured_res.empty and "product_id" in structured_res.columns:
            product_ids = list(structured_res["product_id"])

        semantic_res, semantic_err = user_question(result["semantic_question"], product_ids)

        payload = {
        "structured": structured_res,
        "semantic": semantic_res ,
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
