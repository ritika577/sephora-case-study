import json
from ollama_utils import startup_checks, call_ollama
from config import OLLAMA_MODEL, OLLAMA_GENERATE_URL, EMBED_MODEL
from router import classify_question
from duckdb_connect import process_structured_question
from chroma_connect import user_question
from hybrid_handler import combined_results

_startup_done = False


def _ensure_startup():
    global _startup_done
    if not _startup_done:
        startup_checks(OLLAMA_MODEL)
        startup_checks(EMBED_MODEL)
        _startup_done = True


# =============================================================
# ANSWER SUMMARIZATION
# =============================================================
def _build_summary_prompt(question: str, data_text: str, route: str) -> str:
    """Build a prompt asking the LLM to answer the user's question from the retrieved data."""
    return f"""You are a helpful Sephora product analyst. The user asked a question and
the system retrieved relevant data. Your job is to answer the user's question clearly
and concisely based ONLY on the data provided below.

Rules:
1. Answer in 2-4 clear sentences.
2. Include specific numbers, product names, or brand names from the data.
3. Do not make up information not present in the data.
4. If the data seems insufficient to fully answer, say what you can and note what's missing.
5. Be conversational and helpful, not technical.

User question: {question}

Data source: {route}

Retrieved data:
{data_text}

Answer:""".strip()


def _summarize_structured(question: str, df) -> str:
    """Summarize a DataFrame result into a natural language answer."""
    # Convert DataFrame to a readable text (limit rows to avoid prompt overflow)
    if df is None or df.empty:
        return "No results were found for your question."
    data_text = df.head(20).to_string(index=False)
    prompt = _build_summary_prompt(question, data_text, "database query")
    try:
        return call_ollama(OLLAMA_MODEL, OLLAMA_GENERATE_URL, prompt)
    except Exception:
        return ""


def _parse_semantic_docs(docs) -> list[dict]:
    """Parse ChromaDB JSON document blobs into readable dicts."""
    parsed = []
    if not docs:
        return parsed
    for doc_list in docs:
        for doc_str in doc_list:
            try:
                row = json.loads(doc_str)
                parsed.append({
                    "product_name": row.get("product_name", "Unknown"),
                    "brand_name": row.get("brand_name", "Unknown"),
                    "rating": row.get("rating", "N/A"),
                    "review_text": row.get("review_text", row.get("review_text_clean", "")),
                    "review_title": row.get("review_title", ""),
                })
            except (json.JSONDecodeError, TypeError):
                # If it's not valid JSON, treat as plain text
                parsed.append({
                    "product_name": "Unknown",
                    "brand_name": "Unknown",
                    "rating": "N/A",
                    "review_text": str(doc_str)[:500],
                    "review_title": "",
                })
    return parsed


def _summarize_semantic(question: str, parsed_docs: list[dict]) -> str:
    """Summarize parsed semantic results into a natural language answer."""
    if not parsed_docs:
        return "No relevant reviews were found for your question."
    # Build a readable summary of the reviews for the LLM
    review_texts = []
    for i, doc in enumerate(parsed_docs[:10], 1):
        review_texts.append(
            f"Review {i}: [{doc['brand_name']} - {doc['product_name']}] "
            f"(Rating: {doc['rating']}) {doc['review_text'][:300]}"
        )
    data_text = "\n".join(review_texts)
    prompt = _build_summary_prompt(question, data_text, "review search")
    try:
        return call_ollama(OLLAMA_MODEL, OLLAMA_GENERATE_URL, prompt)
    except Exception:
        return ""


# =============================================================
# MAIN PIPELINE
# =============================================================
def process_user_question(question: str) -> dict:
    if not question or not question.strip():
        return {"route": None, "status": "error", "data": None, "error": "Question is empty."}

    _ensure_startup()
    route = classify_question(question)

    if route == "structured":
        df, err, sql = process_structured_question(question)
        if err:
            return {"route": "structured", "status": "error", "data": None,
                    "error": err, "sql": sql, "answer": None}
        answer = _summarize_structured(question, df)
        return {"route": "structured", "status": "success", "data": df,
                "error": None, "sql": sql, "answer": answer}

    if route == "semantic":
        docs, err = user_question(question)
        if err:
            return {"route": "semantic", "status": "error", "data": None,
                    "error": err, "answer": None, "parsed_docs": None}
        parsed = _parse_semantic_docs(docs)
        answer = _summarize_semantic(question, parsed)
        return {"route": "semantic", "status": "success", "data": docs,
                "error": None, "answer": answer, "parsed_docs": parsed}

    if route == "hybrid":
        result = combined_results(question)
        has_error = result.get("error", {})
        any_err = has_error.get("split") or has_error.get("structured") or has_error.get("semantic")
        status = "error" if has_error.get("split") else ("partial" if any_err else "success")

        # Summarize both parts
        answer_parts = []
        if result.get("structured") is not None and not result["structured"].empty:
            s = _summarize_structured(question, result["structured"])
            if s:
                answer_parts.append(s)
        if result.get("semantic") is not None:
            parsed = _parse_semantic_docs(result["semantic"])
            s = _summarize_semantic(question, parsed)
            if s:
                answer_parts.append(s)
            result["parsed_docs"] = parsed

        answer = "\n\n".join(answer_parts) if answer_parts else None
        return {"route": "hybrid", "status": status, "data": result,
                "error": has_error, "answer": answer}

    return {"route": "unknown", "status": "error", "data": None,
            "error": "Could not classify question.", "answer": None}


if __name__ == "__main__":
    q = "Top 10 brands by average rating"
    output = process_user_question(q)
    print(output)
