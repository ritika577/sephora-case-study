import requests
import re
import json

# constants
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = OLLAMA_BASE_URL + "/api/generate"
OLLAMA_EMBED_URL = OLLAMA_BASE_URL + "/api/embed"
OLLAMA_TAGS_URL = OLLAMA_BASE_URL + "/api/tags"
OLLAMA_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"
MAX_EMBED_CHARS = 8000

# check_ollama_running
def check_ollama_running() -> None:
      try:
          r = requests.get(OLLAMA_BASE_URL, timeout=5)
          r.raise_for_status()
      except requests.exceptions.ConnectionError:
          raise ConnectionError("Ollama is not running. Start it with 'ollama serve'.")
      except requests.exceptions.Timeout:
          raise TimeoutError("Ollama health check timed out.")

# check_model_available      
def check_model_available(model: str) -> None:
      r = requests.get(OLLAMA_TAGS_URL, timeout=5)
      models = [m["name"] for m in r.json().get("models", [])]
      if model not in models:
          raise ValueError(f"Model '{model}' is not pulled. Run 'ollama pull {model}'.")
      
# strip_markdown_fences
def strip_markdown_fences(text: str) -> str:
      match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
      if match:
          return match.group(1)
      return text.strip()

# call_ollama
def call_ollama(model: str, url: str, prompt: str, retries: int = 3) -> str:
      for attempt in range(1, retries + 1):
          try:
              response = requests.post(
                  url,
                  json={"model": model, "prompt": prompt, "stream": False},
                  timeout=120
              )
              response.raise_for_status()
          except requests.exceptions.ConnectionError:
              raise ConnectionError("Ollama is not running. Start it with 'ollama serve'.")
          except requests.exceptions.Timeout:
              raise TimeoutError("Ollama request timed out after 120s.")

          result = response.json().get("response", "").strip()
          if not result:
              if attempt == retries:
                  raise ValueError(f"Ollama returned empty response after {retries} attempts.")
              continue
          return result
      
# call_ollama_json converts json to dict
def call_ollama_json(model: str, url: str, prompt: str, retries: int = 3) -> dict:
      for attempt in range(1, retries + 1):
          raw = call_ollama(model, url, prompt)
          cleaned = strip_markdown_fences(raw)
          try:
              return json.loads(cleaned)
          except json.JSONDecodeError as e:
              print(f"[Attempt {attempt}/{retries}] Invalid JSON: {e}")
              print(f"Raw response: {raw[:200]}")
              if attempt == retries:
                  raise ValueError(f"Failed to get valid JSON after {retries} attempts.")

# embed
def embed(model: str, url: str, text: str, retries: int = 3) -> list:
      text = str(text).strip()
      if not text:
          raise ValueError("Cannot embed empty text.")
      text = text[:MAX_EMBED_CHARS]

      for attempt in range(1, retries + 1):
          try:
              r = requests.post(
                  url,
                  json={"model": model, "input": text},
                  timeout=120
              )
              r.raise_for_status()
              return r.json()["embeddings"][0]
          except requests.exceptions.ConnectionError:
              raise ConnectionError("Ollama is not running. Start it with 'ollama serve'.")
          except requests.exceptions.Timeout:
              raise TimeoutError("Ollama embed request timed out.")
          except Exception as e:
              print(f"[Attempt {attempt}/{retries}] Embed failed: {e}")
              if attempt == retries:
                  raise ValueError(f"Failed to get valid JSON after {retries} attempts.")