# Sequential Task Agent — Investment Analyst
### Problem Statement #4 · Tech Stack: LangChain + RAG + IBM Granite (via Ollama — free & local)

> **100% free. No API keys. No cloud accounts. No billing.**
> IBM Granite runs locally on your machine through [Ollama](https://ollama.com).

---

## Overview

An AI-powered **Sequential Investment Analyst Agent** that automates the full research pipeline — from live market data collection to structured buy/sell recommendations — using **IBM Granite** as the local reasoning model.

```
┌────────────────────────────────────────────────────────────┐
│              6-STEP SEQUENTIAL PIPELINE                     │
├──────┬─────────────────────────────────────────────────────┤
│ S1+2 │ Data Gathering + KPI Extraction                      │
│      │ yfinance API → stock info, financials, price history │
│  S3  │ Financial Ratio Calculation                          │
│      │ P/E, P/B, ROE, D/E, margins, liquidity ratios       │
│  S4  │ Competitor Benchmarking                              │
│      │ Side-by-side peer comparison table                   │
│  S5  │ Market Sentiment Analysis                            │
│      │ Yahoo Finance RSS + News API → Granite NLP           │
│  S6  │ AI Investment Recommendation                         │
│      │ IBM Granite (granite3.3:8b local) → BUY/HOLD/SELL   │
└──────┴─────────────────────────────────────────────────────┘
```

---

## Architecture

```
investment_analyst_agent/
│
├── main.py                  ← CLI entry point  (click + rich)
├── config.py                ← Env-based settings (Ollama URL, model name)
├── llm.py                   ← Local Granite via Ollama (ChatOllama / OllamaLLM)
│
├── tools/
│   └── financial_data.py    ← 6 LangChain @tool wrappers
│                               get_stock_info · get_historical_prices
│                               get_financial_statements · calculate_financial_ratios
│                               get_competitor_data · get_market_news
│
├── rag/
│   └── vector_store.py      ← ChromaDB + HuggingFace embeddings (local)
│                               ingest_documents() · search_reports @tool
│
├── pipeline/
│   ├── agent.py             ← ReAct LangChain agent (full agentic loop)
│   └── analysis.py          ← Structured 6-step direct pipeline (faster)
│
├── ui/
│   └── dashboard.py         ← Streamlit dashboard with Plotly charts
│
└── data/
    └── sample_reports/      ← Drop PDF/TXT reports here for RAG ingestion
```

### Tech Stack — everything free and local
| Component | Technology | Cost |
|-----------|-----------|------|
| **LLM** | IBM Granite `granite3.3:8b` | **Free** |
| **LLM Runtime** | Ollama (local server) | **Free** |
| **LLM Interface** | `langchain-ollama` `ChatOllama` | **Free** |
| **Agent Framework** | LangChain `create_react_agent` | **Free** |
| **RAG** | ChromaDB + `sentence-transformers` | **Free** |
| **Embeddings** | `all-MiniLM-L6-v2` (local) | **Free** |
| **Financial Data** | `yfinance` (public API) | **Free** |
| **News** | Yahoo Finance RSS (+ optional NewsAPI free tier) | **Free** |
| **Dashboard** | Streamlit + Plotly | **Free** |

---

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) installed on your machine
- ~5 GB disk space for the Granite model weights
- Internet access for `yfinance` (live market data)

---

## Quick Start

### 1. Install Ollama and pull IBM Granite

```bash
# Download and install Ollama from https://ollama.com/download
# Then pull the IBM Granite model (one-time, ~5 GB download):
ollama pull granite3.3:8b

# Start the Ollama server (keep this running in a terminal):
ollama serve
```

### 2. Set up the project

```bash
cd investment_analyst_agent

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configure (optional)

The defaults work out of the box. Only create a `.env` if you need to change anything:

```bash
cp .env.example .env
# Defaults — no changes needed for most setups:
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=granite3.3:8b
```

### 4. Run

```bash
# Option A — Streamlit Dashboard (recommended)
streamlit run ui/dashboard.py

# Option B — CLI structured analysis
python main.py analyse AAPL
python main.py analyse TSLA --peers RIVN,GM,F --output tsla_report.json

# Option C — Full ReAct agentic loop (verbose, tool-by-tool)
python main.py agent MSFT --peers GOOGL,AAPL,META

# Option D — Ingest financial reports into RAG
python main.py ingest data/sample_reports/apple_fy2023_summary.txt
```

---

## Usage Details

### Streamlit Dashboard
Open [http://localhost:8501](http://localhost:8501) after running `streamlit run ui/dashboard.py`.

- Enter a ticker (e.g. `AAPL`), optional peers (e.g. `MSFT,GOOGL`)
- Click **Run Analysis** — the 6 pipeline steps execute sequentially
- View: live price chart, KPI cards, financial ratios, competitor bar chart, risk flags, sentiment, AI recommendation

### CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py analyse AAPL` | Fast 6-step pipeline, terminal output |
| `python main.py analyse AAPL --peers MSFT,GOOGL --output report.json` | Save JSON report |
| `python main.py agent AAPL` | Verbose ReAct loop with tool trace |
| `python main.py ingest file.pdf` | Add document to RAG knowledge base |
| `python main.py dashboard` | Launch Streamlit UI |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `granite3.3:8b` | Model to use |
| `NEWS_API_KEY` | *(empty)* | Optional [newsapi.org](https://newsapi.org) free-tier key |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Pipeline Steps

### Step 1+2 — Data Gathering & KPI Extraction
- `get_stock_info` → live price, market cap, 52-week range, analyst consensus
- `get_financial_statements` (income / balance / cash-flow, last 4 years)
- `search_reports` → RAG retrieval from any ingested PDF/TXT documents

### Step 3 — Financial Ratio Calculation
- `calculate_financial_ratios` → 15 ratios: P/E, P/B, EV/EBITDA, PEG, ROE, ROA, margins, current ratio, D/E, FCF yield, dividend yield
- Automatic risk flags (e.g., D/E > 2, current ratio < 1, negative margins, high P/E)

### Step 4 — Competitor Benchmarking
- `get_competitor_data` → side-by-side KPIs for up to 4 peers
- Compares P/E, ROE, net margin, revenue growth, debt/equity

### Step 5 — Market Sentiment Analysis
- `get_market_news` → Yahoo Finance RSS + optional NewsAPI (10 articles)
- Granite NLP → sentiment score (−1.0 to +1.0), BULLISH/NEUTRAL/BEARISH label, 2-3 sentence narrative

### Step 6 — AI Recommendation
- Granite `granite3.3:8b` synthesises all evidence
- Output: investment thesis, STRONG BUY / BUY / HOLD / UNDERWEIGHT / SELL rating, 12-month target, top 3 risks, catalysts, portfolio weight

---

## Alternative Granite Models

You can swap to a smaller or larger model by changing `OLLAMA_MODEL` in `.env`:

| Model | Size | Notes |
|-------|------|-------|
| `granite3.3:2b` | ~2 GB | Faster, less RAM, good for low-resource machines |
| `granite3.3:8b` | ~5 GB | **Recommended** — best balance of quality and speed |
| `granite3.1:8b` | ~5 GB | Previous generation, also works well |

```bash
# Switch models at any time:
ollama pull granite3.3:2b
# Then in .env:
OLLAMA_MODEL=granite3.3:2b
```

---

## Sample Output

```
╭─────────────────────────── Investment Analyst Report ───────────────────────────╮
│ Apple Inc.  STRONG BUY                                                           │
│ AAPL · Technology                                                                │
╰──────────────────────────────────────────────────────────────────────────────────╯

  Key Metrics
  ┌─────────────────────────┬───────────┐
  │ Current Price           │  $189.30  │
  │ Market Cap              │  $2,952B  │
  │ P/E (Trailing)          │  30.5x    │
  │ ROE                     │  147.3%   │
  │ Net Margin              │  25.3%    │
  │ Debt / Equity           │  148.0x   │
  │ Market Sentiment        │  BULLISH  │
  └─────────────────────────┴───────────┘

  ⚠ Risk Flags
    • High P/E — potential overvaluation
    • High leverage — elevated financial risk

  AI-Generated Recommendation (IBM Granite — local)
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ INVESTMENT THESIS                                                            │
  │ Apple continues to demonstrate exceptional profitability with 25% net        │
  │ margins and strong Services segment growth …                                 │
  │                                                                              │
  │ RATING: STRONG BUY                                                           │
  │ TARGET PRICE: $215.00                                                        │
  │ KEY RISKS: China exposure · Valuation premium · AI hardware competition      │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

*Made with IBM Bob · IBM Granite · Ollama · LangChain · 100% free & local*
