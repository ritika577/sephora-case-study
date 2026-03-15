import duckdb
import requests
import re
from typing import Any, Dict, Optional, Tuple

DB_PATH = "sephora.duckdb"
CSV_PATH = "analysis_output/clean_merged.csv"
TABLE_NAME = "sephora"

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

DEFAULT_LIMIT = 50
MAX_SQL_RETRIES = 1


# =========================================================
# DUCKDB SETUP
# =========================================================
con = duckdb.connect(DB_PATH)


# def initialize_database() -> None:
#     """
#     Loads cleaned CSV into DuckDB table.
#     """
#     con.execute(f"""
#         CREATE OR REPLACE TABLE {TABLE_NAME} AS
#         SELECT * FROM read_csv_auto('{CSV_PATH}');
#     """)


# initialize_database()


# =========================================================
# OLLAMA CALL
# =========================================================
def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


# =========================================================
# SCHEMA HELPERS
# =========================================================
def get_schema_info() -> list[tuple]:
    return con.execute(f"DESCRIBE {TABLE_NAME}").fetchall()


def get_schema_text() -> str:
    rows = get_schema_info()
    lines = []

    for row in rows:
        col_name = row[0]
        col_type = row[1]
        lines.append(f"- {col_name}: {col_type}")

    return "\n".join(lines)


# =========================================================
# SQL PROMPT
# =========================================================
def build_sql_prompt(question: str, previous_error: Optional[str] = None) -> str:
    schema_text = get_schema_text()

    prompt = f"""
You are an expert DuckDB SQL generator.

You are working with one table only:
Table name: {TABLE_NAME}

Schema:
{schema_text}

Task:
Convert the user's structured analytics question into one valid DuckDB SQL query.

Strict rules:
1. Output only raw SQL.
2. Do not add markdown.
3. Do not add explanation.
4. Use only the table {TABLE_NAME}.
5. Generate only a read-only query.
6. Allowed query types: SELECT or WITH ... SELECT
7. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, REPLACE, COPY, ATTACH, DETACH, PRAGMA, CALL, EXPORT, IMPORT.
8. Use only columns that exist in the schema.
9. If aggregation is used, use proper GROUP BY.
10. If the user asks for top/bottom, include ORDER BY and LIMIT.
11. Unless the user explicitly asks for all rows, include LIMIT {DEFAULT_LIMIT}.
12. Prefer simple and correct SQL.
13. Do not invent tables or columns.

User question:
{question}
""".strip()

    if previous_error:
        prompt += f"""

Previous SQL validation error:
{previous_error}

Fix the SQL and return only corrected SQL.
""".rstrip()

    return prompt


# =========================================================
# SQL EXTRACTION
# =========================================================
def extract_sql(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    text = raw_text.strip()

    fenced_sql = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced_sql:
        return fenced_sql.group(1).strip().rstrip(";")

    fenced_generic = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if fenced_generic:
        return fenced_generic.group(1).strip().rstrip(";")

    text = re.sub(r"^\s*sql\s*:\s*", "", text, flags=re.IGNORECASE)

    match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if not match:
        return None

    candidate = text[match.start():].strip()

    if ";" in candidate:
        candidate = candidate.split(";", 1)[0].strip()

    return candidate.rstrip(";").strip() or None


# =========================================================
# SQL VALIDATION
# =========================================================
FORBIDDEN_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "COPY",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "CALL",
    "EXPORT",
    "IMPORT",
]


def basic_sql_safety_check(sql: str) -> Tuple[bool, Optional[str]]:
    if not sql or not sql.strip():
        return False, "SQL is empty."

    cleaned = sql.strip()

    if not re.match(r"^\s*(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return False, "Only SELECT or WITH queries are allowed."

    if ";" in cleaned:
        return False, "Multiple SQL statements are not allowed."

    upper_sql = cleaned.upper()

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_sql):
            return False, f"Forbidden SQL keyword detected: {keyword}"

    if TABLE_NAME.lower() not in cleaned.lower():
        return False, f"Query must reference the '{TABLE_NAME}' table."

    return True, None


def explain_validation(sql: str) -> Tuple[bool, Optional[str]]:
    try:
        con.execute(f"EXPLAIN {sql}")
        return True, None
    except Exception as e:
        return False, str(e)


def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
    ok, err = basic_sql_safety_check(sql)
    if not ok:
        return False, err

    ok, err = explain_validation(sql)
    if not ok:
        return False, f"SQL failed EXPLAIN validation: {err}"

    return True, None


# =========================================================
# SQL GENERATION
# =========================================================
def generate_sql(question: str, previous_error: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    try:
        prompt = build_sql_prompt(question, previous_error=previous_error)
        raw_output = call_ollama(prompt)
        sql = extract_sql(raw_output)

        if not sql:
            return None, "Could not extract SQL from LLM output."

        return sql, None

    except Exception as e:
        return None, f"SQL generation failed: {str(e)}"


# =========================================================
# EXECUTION
# =========================================================
def run_query(sql: str):
    return con.execute(sql).fetchdf()


def process_structured_question(question: str) -> Dict[str, Any]:
    """
    This function assumes the router has already decided
    that the question is structured.
    """
    if not question or not question.strip():
        return {
            "status": "error",
            "message": "Question is empty."
        }

    last_error = None
    generated_sql = None

    for attempt in range(MAX_SQL_RETRIES + 1):
        sql, gen_error = generate_sql(question, previous_error=last_error)

        if gen_error:
            return {
                "status": "error",
                "message": gen_error
            }

        generated_sql = sql

        is_valid, validation_error = validate_sql(generated_sql)
        if is_valid:
            break

        last_error = validation_error

        if attempt == MAX_SQL_RETRIES:
            return {
                "status": "error",
                "generated_sql": generated_sql,
                "message": validation_error
            }

    try:
        df = run_query(generated_sql)
        return {
            "status": "success",
            "generated_sql": generated_sql,
            "data": df
        }
    except Exception as e:
        return {
            "status": "error",
            "generated_sql": generated_sql,
            "message": f"Execution failed: {str(e)}"
        }

def sql_answer(question):
    res = process_structured_question(question)
    return run_query(res["generated_sql"])

if __name__ == "__main__":
    result = process_structured_question("Top 10 brands by average rating")
    print(result)