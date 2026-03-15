from chroma_connect import user_question
from duckdb_connect import sql_answer
import json
import requests
import re

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


def call_ollama(prompt: str,retries:int=3) -> str:
    for attempt in range(1,retries+1):
        try:
            response = requests.post(
                OLLAMA_GENERATE_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=60
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Ollama is not running. Start it with 'ollama serve'.")
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out after 60s.")
        
        result = response.json().get("response", "").strip()
        if not result:
            raise ValueError("Ollama returned an empty response.")
        cleaned = strip_markdown_fences(result)
        try:
            result = json.loads(cleaned)
        except Exception as e:
            if attempt == retries:
                    raise ValueError(f"Failed to get valid JSON after {retries} attempts")
        return result
        


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
    result = call_ollama(prompt)
    structured_res = sql_answer(result["structured_question"])
    product_ids=None
    if not structured_res.empty:
        product_ids=structured_res["product_id"]
        product_names=structured_res["product_name"]
        brand_names=structured_res["brand_name"]
        categories=structured_res["primary_category"]
    semantic_res = user_question(result["semantic_question"],product_ids)
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
         "semantic": semantic_res,
         "combined_context": {
            "top_rows": structured_res["rows"][:3] if not structured_res.empty else None,
            "review_snippets": semantic_res[:5],
            "entities": {
                "product_ids": product_ids if not structured_res.empty else None,
                "product_names": product_names if not structured_res.empty else None,
                "brand_names":brand_names if not structured_res.empty else None,
                "categories": categories if not structured_res.empty else None
            }
        }
    }
    return payload


def strip_markdown_fences(text: str) -> str:
      match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
      if match:
          return match.group(1)
      return text.strip()


if __name__ == "__main__":
    result = combined_results("Which moisturizer has the highest rating and what do people complain about?")
    print(result)