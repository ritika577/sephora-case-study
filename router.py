import requests

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


def build_routing_prompt(question: str) -> str:
    return f"""
You are a query router for a Sephora analytics system.

Classify the user's question into exactly one label:

1. structured
- answerable using SQL over structured table data
- examples: counts, averages, filters, top/bottom, grouped comparisons

2. semantic
- needs review text understanding, themes, complaints, sentiment, opinions, summaries

3. hybrid
- needs both SQL and semantic review understanding

Rules:
- Return exactly one word only.
- Allowed outputs: structured, semantic, hybrid
- Do not explain.

User question:
{question}
""".strip()


def classify_question(question: str) -> str:
    """
    Primary classifier using Ollama.
    Small fallback is kept only in case Ollama output is messy.
    """
    prompt = build_routing_prompt(question)

    try:
        raw = call_ollama(prompt).lower().strip()

        if raw in {"structured", "semantic", "hybrid"}:
            return raw

        if "hybrid" in raw:
            return "hybrid"
        if "semantic" in raw:
            return "semantic"
        if "structured" in raw:
            return "structured"

    except Exception:
        pass

    # fallback
    q = question.lower()

    semantic_words = [
        "why", "theme", "themes", "sentiment", "complaint", "complaints",
        "summarize", "summary", "opinion", "opinions", "what do customers say",
        "reviews say", "feedback", "mentioned in reviews"
    ]

    structured_words = [
        "count", "how many", "average", "avg", "sum", "highest", "lowest",
        "top", "bottom", "most", "least", "show", "list", "compare",
        "distribution", "breakdown", "filter", "above", "below", "between",
        "price", "rating", "brand", "category", "online only",
        "sephora exclusive", "out of stock"
    ]

    has_structured = any(word in q for word in structured_words)
    has_semantic = any(word in q for word in semantic_words)

    if has_structured and has_semantic:
        return "hybrid"
    if has_structured:
        return "structured"
    return "semantic"