"""
pipeline/analysis.py
─────────────────────────────────────────────────────────────────────────────
Structured analysis helpers used by both the CLI and Streamlit dashboard.
These functions wrap the raw tool outputs and call Granite via the direct API
for analysis steps that do NOT need the agent loop (faster, cheaper).

Exposed functions:
  - extract_kpis(symbol)         → dict of key financial KPIs
  - analyse_sentiment(news_json) → sentiment score + summary string
  - generate_recommendation(ctx) → structured recommendation text (Granite call)
  - build_full_analysis(symbol)  → runs all steps, returns AnalysisResult
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from tools.financial_data import (
    calculate_financial_ratios,
    get_competitor_data,
    get_financial_statements,
    get_market_news,
    get_stock_info,
)
from llm import generate_text

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    symbol: str
    company_name: str = ""
    sector: str = ""
    current_price: float = 0.0
    market_cap_b: float = 0.0
    kpis: dict[str, Any] = field(default_factory=dict)
    ratios: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0        # -1.0 (bearish) … +1.0 (bullish)
    sentiment_label: str = "NEUTRAL"
    news_summary: str = ""
    competitor_table: list[dict] = field(default_factory=list)
    recommendation: str = ""
    rating: str = "HOLD"
    target_price: float | None = None
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Step helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_kpis(symbol: str) -> dict[str, Any]:
    """
    Step 1+2: Gather raw data and extract key financial KPIs.
    Returns a flat dict with the most investable figures.
    """
    kpis: dict[str, Any] = {}

    # Live snapshot
    try:
        raw_info = json.loads(get_stock_info.invoke(symbol))
        kpis.update({
            "company_name": raw_info.get("longName", symbol),
            "sector": raw_info.get("sector", ""),
            "industry": raw_info.get("industry", ""),
            "current_price": raw_info.get("currentPrice"),
            "previous_close": raw_info.get("previousClose"),
            "market_cap_b": round((raw_info.get("marketCap") or 0) / 1e9, 2),
            "52w_high": raw_info.get("fiftyTwoWeekHigh"),
            "52w_low": raw_info.get("fiftyTwoWeekLow"),
            "analyst_target": raw_info.get("targetMeanPrice"),
            "analyst_recommendation": raw_info.get("recommendationKey"),
            "beta": raw_info.get("beta"),
            "dividend_yield": raw_info.get("dividendYield"),
            # NOTE: this was previously missing, which is why the dashboard's
            # "Data as of ..." caption always showed "unknown" — get_stock_info
            # sets a timestamp on the snapshot, but it was never copied here.
            "timestamp": raw_info.get("timestamp"),
        })
    except Exception as exc:
        logger.warning("KPI extraction (info) failed: %s", exc)
        kpis["info_error"] = str(exc)

    # Income statement KPIs
    try:
        income_raw = get_financial_statements.invoke(
            {"symbol": symbol, "statement": "income"}
        )
        income = json.loads(income_raw)
        if isinstance(income, list) and income:
            # income is list of {metric: ..., col1: val, col2: val, ...}
            # Convert to {metric: latest_value}
            latest_col = None
            for row in income:
                keys = [k for k in row if k != "index"]
                if not latest_col and keys:
                    latest_col = sorted(keys, reverse=True)[0]
            if latest_col:
                for row in income:
                    kpis[str(row.get("index", ""))] = row.get(latest_col)
    except Exception as exc:
        logger.warning("KPI extraction (income) failed: %s", exc)

    return kpis


def _strip_markdown_emphasis(text: str) -> str:
    """Remove **bold**, __bold__, *italic* wrappers local LLMs sometimes add
    around structured labels (e.g. '**SCORE:** 0.6') so label matching is
    not fooled by formatting."""
    return re.sub(r"[\*_]+", "", text).strip()


def analyse_sentiment(news_json: str) -> tuple[float, str, str]:
    """
    Step 5: Analyse market sentiment from a news JSON string.

    Returns:
        (score, label, summary_text)
        score: float from -1.0 (very bearish) to +1.0 (very bullish)
        label: 'BULLISH' | 'NEUTRAL' | 'BEARISH'
        summary_text: 2-3 sentence AI summary
    """
    try:
        articles = json.loads(news_json)
    except (json.JSONDecodeError, TypeError):
        return 0.0, "NEUTRAL", "Unable to parse news data."

    if not articles or (isinstance(articles, dict) and "message" in articles):
        return 0.0, "NEUTRAL", "No recent news available for sentiment analysis."

    if isinstance(articles, list):
        headlines = "\n".join(
            f"- {a.get('title', '')} ({a.get('source', '')})"
            for a in articles[:10]
        )
    else:
        headlines = str(articles)

    prompt = f"""You are a financial sentiment analyst. Analyse these news headlines and provide:
1. A sentiment score from -1.0 (very bearish) to +1.0 (very bullish) as a plain number.
2. A label: BULLISH, NEUTRAL, or BEARISH.
3. A 2-3 sentence summary of the market narrative.

Format your response EXACTLY as (no markdown formatting, no bold, plain text only):
SCORE: <number>
LABEL: <BULLISH|NEUTRAL|BEARISH>
SUMMARY: <2-3 sentence summary>

Headlines:
{headlines}
"""
    response = generate_text(prompt, max_tokens=200)

    # Parse the structured response. Local models sometimes wrap labels in
    # markdown emphasis (**SCORE:**) or vary casing (Score:) — both are
    # normalised before matching so a formatting quirk doesn't silently
    # drop the value back to its default.
    score = 0.0
    label = "NEUTRAL"
    summary = response

    for raw_line in response.splitlines():
        line = _strip_markdown_emphasis(raw_line)
        upper = line.upper()
        if upper.startswith("SCORE:"):
            value_str = line.split(":", 1)[1].strip()
            value_str = re.sub(r"[^0-9.\-+]", "", value_str)  # keep numeric chars only
            try:
                score = float(value_str)
                score = max(-1.0, min(1.0, score))
            except ValueError:
                pass
        elif upper.startswith("LABEL:"):
            lbl = line.split(":", 1)[1].strip().upper()
            if lbl in ("BULLISH", "NEUTRAL", "BEARISH"):
                label = lbl
        elif upper.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    return score, label, summary


def _extract_target_price(report: str) -> float | None:
    """
    Pull the 12-month target price out of the AI-generated report.

    The report format puts the dollar figure on its own line *after* the
    'TARGET PRICE' heading (e.g. '3. TARGET PRICE (12-month, USD)\n\n$429.46'),
    not on the heading line itself. Scanning line-by-line for any line that
    merely *contains* the words 'TARGET PRICE' — as this used to do — matches
    the heading line and greedily extracts its section number ('3.') instead
    of the real price on the following line.

    Instead: isolate the TARGET PRICE section (from its heading up to the
    next numbered heading), then look for a dollar-formatted number within
    just that section.
    """
    section_match = re.search(
        r"TARGET PRICE.*?(?=\n\s*\d+\.\s+[A-Z]|\Z)",
        report,
        re.IGNORECASE | re.DOTALL,
    )
    search_text = section_match.group(0) if section_match else report

    # Prefer a dollar-sign figure with cents (the realistic price format).
    candidates = re.findall(r"\$\s*([\d,]+\.\d{1,2})", search_text)
    if not candidates:
        # Fall back to a dollar-sign figure without cents.
        candidates = re.findall(r"\$\s*([\d,]+)", search_text)

    for m in candidates:
        try:
            val = float(m.replace(",", ""))
            if 1 < val < 100000:   # Sanity check
                return val
        except ValueError:
            continue
    return None


def generate_recommendation(analysis_context: str) -> tuple[str, str, float | None]:
    """
    Step 6: Ask Granite to produce a final investment recommendation.

    Args:
        analysis_context: JSON / text summary of all prior analysis steps.

    Returns:
        (full_report_text, rating, target_price)
    """
    prompt = f"""You are a senior investment analyst. Based on the following analysis,
produce a professional investment report with these sections:

1. INVESTMENT THESIS (3-5 sentences)
2. RATING: one of STRONG BUY | BUY | HOLD | UNDERWEIGHT | SELL
3. TARGET PRICE (12-month, USD, as a number)
4. KEY RISKS (top 3 bullet points)
5. CATALYSTS TO WATCH (top 3 bullet points)
6. PORTFOLIO WEIGHTING SUGGESTION (e.g., 3-5% core holding, tactical position)

Base your analysis STRICTLY on the provided data. Acknowledge uncertainty where data is missing.

Analysis Data:
{analysis_context}

Begin the report now:"""

    report = generate_text(prompt, max_tokens=600)

    # Extract structured fields from the report
    rating = "HOLD"

    for line in report.splitlines():
        upper = line.upper()
        if "STRONG BUY" in upper:
            rating = "STRONG BUY"
        elif "UNDERWEIGHT" in upper:
            rating = "UNDERWEIGHT"
        elif "BUY" in upper and "STRONG" not in upper:
            rating = "BUY"
        elif "SELL" in upper and "UNDERWEIGHT" not in upper:
            rating = "SELL"
        elif "HOLD" in upper:
            rating = "HOLD"

    target_price = _extract_target_price(report)

    return report, rating, target_price


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: runs all steps sequentially and returns AnalysisResult
# ─────────────────────────────────────────────────────────────────────────────

def build_full_analysis(symbol: str, competitors: str = "") -> AnalysisResult:
    """
    Run the complete 6-step sequential analysis pipeline without the
    agent loop (faster path used by the Streamlit dashboard for structured output).

    Steps:
      1 & 2 — Data gathering + KPI extraction
          3 — Financial ratio calculation
          4 — Competitor benchmarking
          5 — Market sentiment analysis
          6 — AI-generated recommendation (Granite)

    Args:
        symbol: Ticker symbol.
        competitors: Optional comma-separated peer tickers.

    Returns:
        AnalysisResult dataclass populated with all findings.
    """
    symbol = symbol.upper().strip()
    result = AnalysisResult(symbol=symbol)

    logger.info("[STEP 1+2] Data gathering & KPI extraction for %s", symbol)
    try:
        kpis = extract_kpis(symbol)
        result.kpis = kpis
        result.company_name = kpis.get("company_name", symbol)
        result.sector = kpis.get("sector", "")
        result.current_price = float(kpis.get("current_price") or 0)
        result.market_cap_b = float(kpis.get("market_cap_b") or 0)
    except Exception as exc:
        logger.error("Step 1+2 failed: %s", exc)
        result.error = str(exc)
        return result

    logger.info("[STEP 3] Financial ratio calculation for %s", symbol)
    try:
        ratio_raw = calculate_financial_ratios.invoke(symbol)
        ratios = json.loads(ratio_raw)
        result.ratios = ratios
        result.risk_flags = ratios.get("risk_flags", [])
    except Exception as exc:
        logger.warning("Step 3 failed: %s", exc)

    logger.info("[STEP 4] Competitor benchmarking for %s", symbol)
    try:
        comp_raw = get_competitor_data.invoke(
            {"symbol": symbol, "competitors": competitors}
        )
        parsed_comp = json.loads(comp_raw)
        # Defense in depth: get_competitor_data should always return a list,
        # but if it ever comes back as an error object instead (e.g. an
        # older/unpatched tools/financial_data.py), don't let a dict get
        # treated as the competitor table — iterating a dict yields its
        # string keys, which crashes downstream `.get()` calls.
        result.competitor_table = parsed_comp if isinstance(parsed_comp, list) else []
    except Exception as exc:
        logger.warning("Step 4 failed: %s", exc)
        result.competitor_table = []

    logger.info("[STEP 5] Market sentiment analysis for %s", symbol)
    try:
        news_raw = get_market_news.invoke({"query": symbol, "max_articles": 10})
        score, label, summary = analyse_sentiment(news_raw)
        result.sentiment_score = score
        result.sentiment_label = label
        result.news_summary = summary
    except Exception as exc:
        logger.warning("Step 5 failed: %s", exc)

    logger.info("[STEP 6] Generating AI recommendation for %s", symbol)
    try:
        # Build a concise context string for Granite
        ctx_dict = {
            "symbol": symbol,
            "company": result.company_name,
            "sector": result.sector,
            "current_price": result.current_price,
            "market_cap_B": result.market_cap_b,
            "analyst_consensus": result.kpis.get("analyst_recommendation"),
            "analyst_target": result.kpis.get("analyst_target"),
            "ratios": {
                k: v for k, v in result.ratios.items()
                if k not in ("symbol", "risk_flags") and v is not None
            },
            "risk_flags": result.risk_flags,
            "sentiment": result.sentiment_label,
            "news_summary": result.news_summary,
            "competitor_comparison": result.competitor_table[:3],
        }
        context_str = json.dumps(ctx_dict, indent=2, default=str)
        report, rating, target_price = generate_recommendation(context_str)
        result.recommendation = report
        result.rating = rating
        result.target_price = target_price
    except Exception as exc:
        logger.error("Step 6 failed: %s", exc)
        result.recommendation = f"Recommendation generation failed: {exc}"

    return result