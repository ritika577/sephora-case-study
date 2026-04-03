from ollama_utils import startup_checks
from config import OLLAMA_MODEL, EMBED_MODEL
from router import classify_question
from duckdb_connect import process_structured_question
from chroma_connect import user_question
from hybrid_handler import combined_results


def process_user_question(question: str) -> dict:
    if not question or not question.strip():
        return {"route": None, "status": "error", "data": None, "error": "Question is empty."}

    startup_checks(OLLAMA_MODEL)
    startup_checks(EMBED_MODEL)
    route = classify_question(question)

    if route == "structured":
        df, err = process_structured_question(question)
        if err:
            return {"route": "structured", "status": "error", "data": None, "error": err}
        return {"route": "structured", "status": "success", "data": df, "error": None}

    if route == "semantic":
        docs, err = user_question(question)
        if err:
            return {"route": "semantic", "status": "error", "data": None, "error": err}
        return {"route": "semantic", "status": "success", "data": docs, "error": None}

    if route == "hybrid":
        result = combined_results(question)
        has_error = result.get("error", {})
        any_err = has_error.get("split") or has_error.get("structured") or has_error.get("semantic")
        status = "error" if has_error.get("split") else ("partial" if any_err else "success")
        return {"route": "hybrid", "status": status, "data": result, "error": has_error}

    return {"route": "unknown", "status": "error", "data": None, "error": "Could not classify question."}


if __name__ == "__main__":
    q = "Top 10 brands by average rating"
    output = process_user_question(q)
    print(output)