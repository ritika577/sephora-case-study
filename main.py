from router import classify_question
from duckdb_connect import process_structured_question
from chroma_connect import user_question
from hybrid_handler import combined_results


def process_user_question(question: str):
    route = classify_question(question)

    if route == "structured":
        return {
            "route": "structured",
            "result": process_structured_question(question)
        }

    if route == "semantic":
        return {
            "route": "semantic",
            "result": user_question(question)
        }

    if route == "hybrid":
        return {
            "route": "hybrid",
            "result": combined_results(question)
        }

    return {
        "route": "unknown",
        "result": {
            "status": "error",
            "message": "Could not classify question."
        }
    }


if __name__ == "__main__":
    q = "Top 10 brands by average rating"
    output = process_user_question(q)
    print(output)