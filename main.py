from router import classify_question
from duckdb_connect import sql_answer
from chroma_connect import user_question
# from hybrid_handler import process_hybrid_question


def process_user_question(question: str):
    route = classify_question(question)

    if route == "structured":
        return {
            "route": "structured",
            "result": sql_answer(question)
        }

    if route == "semantic":
        return {
            "route": "semantic",
            "result": user_question(question)
        }

    if route == "hybrid":
        return {
            "route": "hybrid",
            "result": {
                "status": "todo",
                "message": "Send this question to hybrid pipeline."
            }
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