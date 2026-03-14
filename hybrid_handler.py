from chroma_connect import user_question
from duckdb_connect import sql_answer
import json
import requests
from router import classify_question

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["response"].strip()


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

if __name__ == "__main__":
    prompt = build_hybrid_split_prompt("how many products you have of Algenist and name those products?")
    result = call_ollama(prompt)
    print(result)
    result_dict=json.loads(result)
    print("result_dict: ",result_dict)
    structured_res = sql_answer(result_dict["structured_question"])
    print("structured_res", structured_res)
    semantic_res = user_question(result_dict["semantic_question"])
    print("semantic_res", semantic_res)



    
