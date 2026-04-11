# Sephora Skincare Analysis Dashboard

An interactive data analysis dashboard for Sephora skincare products, built with Streamlit and Plotly. It combines pre-computed analytics with an AI-powered natural language query system (Ask AI) that lets you explore ~1,700 products across 140 brands using plain English.

## Features

- **Overview** — Key metrics, most loved products, and brand popularity (with normalized loves-per-product view)
- **Brand Analysis** — Bubble chart comparing brand catalog size vs product quality (median rating and loves)
- **Price Analysis** — Brand pricing strategies and price tier distribution
- **Sentiment Analysis** — Review sentiment breakdown, brand sentiment ranking, and recommendation rates
- **Ask AI** — Ask questions in plain English. Routes queries to SQL (structured), review search (semantic), or both (hybrid)

## Tech Stack

- **Dashboard**: Streamlit + Plotly
- **Data Processing**: Pandas, NumPy
- **Databases**: DuckDB (structured queries), ChromaDB (semantic search)
- **NLP**: VADER Sentiment Analysis, Ollama (local LLM for SQL generation, query routing, and answer summarization)
- **LLM Models**: Qwen 2.5 7B (generation), Nomic Embed Text (embeddings) — runs locally via Ollama

## Project Structure

```
├── config.py              # All configuration (paths, model names, DB settings)
├── ingest.py              # Data ingestion pipeline (clean → DuckDB → ChromaDB)
├── analysis.py            # Pre-compute analytics CSVs for the dashboard
├── data_cleaning.py       # Text cleaning, price normalization, size parsing
├── streamlit.py           # Dashboard (all pages)
├── main.py                # Ask AI pipeline (routing → query → summarization)
├── router.py              # Question classifier (structured/semantic/hybrid)
├── duckdb_connect.py      # SQL generation, validation, and execution
├── chroma_connect.py      # Semantic search via ChromaDB
├── hybrid_handler.py      # Combined structured + semantic queries
├── ollama_utils.py        # Ollama API helpers (generate, embed, health checks)
├── tests/test_core.py     # Unit tests
├── data/raw/              # Raw CSV files (product_info.csv, reviews_*.csv)
├── analysis_output/       # Generated CSVs consumed by the dashboard
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### Step 1: Clone and set up the environment

```bash
git clone <repo-url>
cd sephora-case-study

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Step 2: Download the required Ollama models

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Make sure Ollama is running:

```bash
ollama serve
```

### Step 3: Add the raw data

Place the following CSV files in `data/raw/`:

- `product_info.csv` — Product catalog (product details, prices, categories)
- `reviews_0-250.csv`, `reviews_250-500.csv`, `reviews_500-750.csv`, `reviews_750-1250.csv`, `reviews_1250-end.csv` — Review data split across files

### Step 4: Run the ingestion pipeline

This merges raw CSVs, cleans the data, and loads it into DuckDB and ChromaDB:

```bash
python ingest.py
```

You can also run individual steps:

```bash
python ingest.py --clean          # Merge and clean only
python ingest.py --duckdb         # Reload DuckDB from existing clean CSV
python ingest.py --chroma         # Reload ChromaDB from existing clean CSV
```

### Step 5: Run the analysis

This generates the pre-computed CSV files used by the dashboard:

```bash
python analysis.py
```

### Step 6: Launch the dashboard

```bash
streamlit run streamlit.py
```

The dashboard will open at `http://localhost:8501`.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:7b` | LLM for SQL generation and summarization |
| `EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model for semantic search |
| `DB_PATH` | `sephora.duckdb` | DuckDB database file |
| `CHROMA_PATH` | `chroma_store/` | ChromaDB storage directory |
| `DEFAULT_LIMIT` | `50` | Default row limit for SQL queries |
| `MAX_SQL_RETRIES` | `3` | Retry attempts for SQL generation |

## Notes

- The dataset covers **Skincare products only**. Questions about Makeup, Fragrance, Hair, etc. cannot be answered.
- Ask AI requires Ollama to be running with both models pulled.
- The dashboard pages work without Ollama — only Ask AI needs it.
