"""
pipeline/agent.py
─────────────────────────────────────────────────────────────────────────────
Sequential Task Agent — Investment Analyst
─────────────────────────────────────────────────────────────────────────────
Implements a LangChain agent backed by IBM Granite running locally via Ollama.
Uses LangChain 1.4+ `create_agent` API (compiled state graph).

Sequential pipeline (enforced via system prompt + ordered sub-prompts):
  Step 1 → Data Gathering   : stock info + price history + financial statements
  Step 2 → KPI Extraction   : parse key metrics from raw data
  Step 3 → Ratio Calculation: compute P/E, ROE, margins, leverage, etc.
  Step 4 → Competitor Bench.: compare against industry peers
  Step 5 → Market Sentiment : analyse recent news + analyst ratings
  Step 6 → Report Generation: structured investment recommendation
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from llm import get_granite_llm
from rag.vector_store import search_reports
from tools.financial_data import ALL_FINANCIAL_TOOLS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — investment analyst persona
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Investment Analyst AI powered by IBM Granite (running locally via Ollama).
You perform rigorous, data-driven analysis and produce actionable investment recommendations.

## Analysis Pipeline (follow this EXACT sequence)
You MUST complete all 6 steps before producing the final report. Do not skip any step.

Step 1 — DATA GATHERING
  • Use get_stock_info to get the live snapshot for the target ticker.
  • Use get_historical_prices with period='1y' to get the price trend.
  • Use get_financial_statements for 'income', 'balance', and 'cashflow'.
  • Use search_reports to retrieve any relevant analyst reports or filings.

Step 2 — KPI EXTRACTION
  • From the financial statements, identify: Revenue, Gross Profit, Operating Income,
    Net Income, Total Assets, Total Debt, Cash & Equivalents, Free Cash Flow.
  • Note year-over-year changes for each KPI.

Step 3 — RATIO CALCULATION
  • Use calculate_financial_ratios to get the full ratio set.
  • Compare ratios against typical sector benchmarks.
  • Flag any ratios outside safe ranges (e.g., D/E > 2, current ratio < 1).

Step 4 — COMPETITOR BENCHMARKING
  • Use get_competitor_data with 3-4 industry peers.
  • Identify where the target company leads or lags peers on valuation, margins, growth.

Step 5 — MARKET SENTIMENT ANALYSIS
  • Use get_market_news to fetch the 10 most recent articles.
  • Summarise the overall sentiment (bullish / neutral / bearish).
  • Note any material events (earnings beats/misses, lawsuits, product launches).

Step 6 — GENERATE INVESTMENT REPORT
  • Based on ALL above evidence, produce a structured recommendation:
    - Investment thesis (3-5 sentences)
    - Rating: STRONG BUY | BUY | HOLD | UNDERWEIGHT | SELL
    - Target price (12-month horizon estimate)
    - Key risks (top 3)
    - Catalysts to watch
    - Suggested portfolio weighting

## Rules
- Always cite the data source for each claim.
- Express uncertainty explicitly when data is incomplete.
- Never fabricate financial figures.
- Keep all responses factual and professional.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

def build_investment_agent():
    """
    Build and return a compiled LangChain 1.4 agent graph.

    The agent has access to:
      - All 6 financial data tools (yfinance + News API)
      - The RAG search_reports tool (ChromaDB knowledge base)
    """
    llm = get_granite_llm()
    all_tools = ALL_FINANCIAL_TOOLS + [search_reports]

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
    )

    logger.info(
        "Investment agent built with %d tools on model: %s",
        len(all_tools),
        settings.OLLAMA_MODEL,
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# High-level run function
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    symbol: str,
    competitors: str = "",
    extra_context: str = "",
) -> dict[str, Any]:
    """
    Run the full 6-step investment analysis for a given ticker.

    Args:
        symbol: Stock ticker, e.g. 'AAPL'.
        competitors: Optional comma-separated peers, e.g. 'MSFT,GOOGL'.
        extra_context: Any additional instructions or focus areas.

    Returns:
        Dict with keys:
          - symbol
          - report   (final text from the agent's last message)
          - steps    (list of tool-call dicts)
          - error    (None or exception string)
    """
    agent = build_investment_agent()

    user_request = (
        f"Perform a complete 6-step investment analysis for stock ticker: **{symbol.upper()}**.\n"
    )
    if competitors:
        user_request += f"Benchmark against these peers: {competitors}.\n"
    if extra_context:
        user_request += f"Additional focus: {extra_context}\n"
    user_request += "\nFollow all 6 pipeline steps and conclude with a structured investment report."

    result: dict[str, Any] = {
        "symbol": symbol.upper(),
        "report": "",
        "steps": [],
        "error": None,
    }

    try:
        response = agent.invoke({"messages": [HumanMessage(content=user_request)]})

        # Extract final text answer from the last AI message
        messages = response.get("messages", [])
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if content and not getattr(msg, "tool_calls", None):
                result["report"] = content
                break

        # Extract tool call steps from message history
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    result["steps"].append({
                        "tool": tc.get("name", ""),
                        "input": str(tc.get("args", ""))[:300],
                    })

    except Exception as exc:
        logger.error("Agent run failed for %s: %s", symbol, exc, exc_info=True)
        result["error"] = str(exc)

    return result
