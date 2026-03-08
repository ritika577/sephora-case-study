import requests
import pandas as pd

def ask_ollama(prompt: str, model: str = "qwen2.5:7b") -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]

def answer_question(df: pd.DataFrame, user_q: str, model: str = "qwen2.5:7b") -> str:
    print("kjhgfdsa", df.dtypes.astype(str).to_dict())
    schema = f"""
Columns: {list(df.columns)}
Dtypes: {df.dtypes.astype(str).to_dict()}
Sample rows:
{df.sample(min(5, len(df)), random_state=42).to_csv(index=False)}
"""
    prompt = f"{schema}\n\nUser question: {user_q}\nAnswer briefly using only available columns."
    return ask_ollama(prompt, model=model)

df = pd.read_csv("analysis_output/clean_merged.csv")
output = answer_question(df,"how many products are there in sephora, provide meexact count?")
print("final result", output)