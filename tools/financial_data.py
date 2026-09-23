"""
tools/financial_data.py
─────────────────────────────────────────────────────────────────────────────
LangChain Tool wrappers that pull live financial data via yfinance and
public RSS / News API feeds.  All tools are decorated with @tool so they can
be registered directly on any LangChain agent.

Tools exposed:
  - get_stock_info          : live quote + company overview
  - get_historical_prices   : OHLCV history as JSON
  - get_financial_statements: income / balance / cash-flow statements
  - get_market_news         : recent news headlines from RSS + optional News API
  - calculate_financial_ratios : P/E, P/B, ROE, Debt/Equity, current ratio …
  - get_competitor_data     : side-by-side KPIs for a peer group
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import feedparser
import pandas as pd
import requests
import yfinance as yf
from langchain.tools import tool

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value: Any, decimals: int = 4) -> float | None:
    """Coerce a value to float, returning None on failure."""
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol.upper().strip())


def _to_json(obj: Any) -> str:
    """Serialise pandas objects and standard types to a JSON string."""
    if isinstance(obj, pd.DataFrame):
        obj = obj.reset_index()
        obj.columns = [str(c) for c in obj.columns]
        return obj.to_json(orient="records", date_format="iso")
    if isinstance(obj, pd.Series):
        return obj.to_json(date_format="iso")
    return json.dumps(obj, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# LangChain Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_stock_info(symbol: str) -> str:
    """
    Fetch a real-time snapshot for a stock ticker.

    Args:
        symbol: Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'IBM'.

    Returns:
        JSON string with company overview, sector, current price, market cap,
        52-week high/low, average volume, and analyst target price.
    """
    try:
        info = _ticker(symbol).info
        fields = [
            "longName", "sector", "industry", "country",
            "currentPrice", "previousClose", "open", "dayHigh", "dayLow",
            "volume", "averageVolume",
            "marketCap", "enterpriseValue",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "targetMeanPrice", "recommendationKey",
            "trailingPE", "forwardPE", "priceToBook",
            "returnOnEquity", "returnOnAssets",
            "revenueGrowth", "earningsGrowth",
            "totalDebt", "totalCash", "freeCashflow",
            "dividendYield", "payoutRatio",
            "beta",
        ]
        snapshot = {k: info.get(k) for k in fields}
        snapshot["symbol"] = symbol.upper()
        snapshot["timestamp"] = datetime.utcnow().isoformat()
        return json.dumps(snapshot, default=str)
    except Exception as exc:
        logger.error("get_stock_info failed for %s: %s", symbol, exc)
        return json.dumps({"error": str(exc), "symbol": symbol})


@tool
def get_historical_prices(symbol: str, period: str = "1y") -> str:
    """
    Retrieve historical OHLCV price data for a ticker.

    Args:
        symbol: Ticker symbol, e.g. 'TSLA'.
        period: Time window — '1mo', '3mo', '6mo', '1y', '2y', '5y'. Default '1y'.

    Returns:
        JSON array of {Date, Open, High, Low, Close, Volume} records.
    """
    try:
        df = _ticker(symbol).history(period=period)
        if df.empty:
            return json.dumps({"error": "No price data returned", "symbol": symbol})
        df = df[["Open", "High", "Low", "Close", "Volume"]].round(4)
        # Keep weekly samples for large periods to stay within token limits
        if period in ("2y", "5y"):
            df = df.resample("W").last()
        df.index = df.index.strftime("%Y-%m-%d")
        return _to_json(df.reset_index().rename(columns={"index": "Date"}))
    except Exception as exc:
        logger.error("get_historical_prices failed for %s: %s", symbol, exc)
        return json.dumps({"error": str(exc), "symbol": symbol})


@tool
def get_financial_statements(symbol: str, statement: str = "income") -> str:
    """
    Retrieve annual financial statements for a company.

    Args:
        symbol: Ticker symbol.
        statement: One of 'income', 'balance', 'cashflow'. Default 'income'.

    Returns:
        JSON representation of the last 4 years of annual figures.
    """
    try:
        tk = _ticker(symbol)
        stmt_map = {
            "income": tk.financials,
            "balance": tk.balance_sheet,
            "cashflow": tk.cashflow,
        }
        df = stmt_map.get(statement.lower())
        if df is None:
            return json.dumps({"error": f"Unknown statement type: {statement}"})
        if df is not None and not df.empty:
            df.columns = pd.to_datetime(df.columns).strftime("%Y-%m-%d")
            return _to_json(df)
        return json.dumps({"error": "No data", "symbol": symbol})
    except Exception as exc:
        logger.error("get_financial_statements failed for %s: %s", symbol, exc)
        return json.dumps({"error": str(exc)})


@tool
def calculate_financial_ratios(symbol: str) -> str:
    """
    Compute key financial ratios for a ticker from live data.

    Ratios calculated:
      P/E (trailing & forward), P/B, EV/EBITDA, Debt-to-Equity,
      Current Ratio, Quick Ratio, ROE, ROA, Gross Margin,
      Operating Margin, Net Margin, Free Cash Flow Yield,
      Dividend Yield, PEG Ratio.

    Args:
        symbol: Ticker symbol.

    Returns:
        JSON object with all computed ratios and an interpretation note.
    """
    try:
        info = _ticker(symbol).info

        ratios: dict[str, Any] = {
            "symbol": symbol.upper(),
            # Valuation
            "pe_trailing": _safe_float(info.get("trailingPE")),
            "pe_forward": _safe_float(info.get("forwardPE")),
            "price_to_book": _safe_float(info.get("priceToBook")),
            "ev_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
            "peg_ratio": _safe_float(info.get("pegRatio")),
            # Profitability
            "roe": _safe_float(info.get("returnOnEquity")),
            "roa": _safe_float(info.get("returnOnAssets")),
            "gross_margin": _safe_float(info.get("grossMargins")),
            "operating_margin": _safe_float(info.get("operatingMargins")),
            "net_margin": _safe_float(info.get("profitMargins")),
            # Liquidity / Leverage
            "current_ratio": _safe_float(info.get("currentRatio")),
            "quick_ratio": _safe_float(info.get("quickRatio")),
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            # Growth
            "revenue_growth_yoy": _safe_float(info.get("revenueGrowth")),
            "earnings_growth_yoy": _safe_float(info.get("earningsGrowth")),
            # Yield
            "dividend_yield": _safe_float(info.get("dividendYield")),
            "free_cashflow_yield": None,
        }

        # Compute FCF yield manually
        market_cap = info.get("marketCap")
        fcf = info.get("freeCashflow")
        if market_cap and fcf and market_cap > 0:
            ratios["free_cashflow_yield"] = _safe_float(fcf / market_cap)

        # Simple flag system
        flags: list[str] = []
        if ratios["pe_trailing"] and ratios["pe_trailing"] > 30:
            flags.append("High P/E — potential overvaluation")
        if ratios["debt_to_equity"] and ratios["debt_to_equity"] > 200:
            flags.append("High leverage — elevated financial risk")
        if ratios["net_margin"] and ratios["net_margin"] < 0:
            flags.append("Negative net margin — unprofitable")
        if ratios["revenue_growth_yoy"] and ratios["revenue_growth_yoy"] < -0.05:
            flags.append("Declining revenues — growth concern")
        if ratios["current_ratio"] and ratios["current_ratio"] < 1:
            flags.append("Current ratio < 1 — potential liquidity risk")
        ratios["risk_flags"] = flags

        return json.dumps(ratios, default=str)
    except Exception as exc:
        logger.error("calculate_financial_ratios failed for %s: %s", symbol, exc)
        return json.dumps({"error": str(exc)})


@tool
def get_competitor_data(symbol: str, competitors: str = "") -> str:
    """
    Retrieve and compare key financial metrics for a stock and its peers.

    Args:
        symbol: Primary ticker symbol.
        competitors: Comma-separated peer tickers, e.g. 'MSFT,GOOGL,META'.
                     If empty, yfinance recommendations are used.

    Returns:
        JSON array of per-company KPI rows for benchmarking. Always a JSON
        *array* (possibly empty) — never an error object — so callers can
        safely iterate the result without a type check.
    """
    rows: list[dict] = []
    try:
        peer_list = [s.strip().upper() for s in competitors.split(",") if s.strip()]
        if not peer_list:
            # Fall back to yfinance analyst recommendations list
            try:
                recs = _ticker(symbol).recommendations
                if recs is not None and not recs.empty:
                    peer_list = list(recs["symbol"].dropna().unique()[:4])
            except Exception as exc:
                logger.warning("get_competitor_data: peer lookup failed for %s: %s", symbol, exc)

        all_symbols = [symbol.upper()] + peer_list[:4]

        for sym in all_symbols:
            # Each peer is fetched independently: one bad/rate-limited ticker
            # must not blank out the whole comparison table. Previously this
            # whole loop shared one try/except, so a single failure fell
            # through to the outer handler and returned {"error": ...} —
            # which the dashboard then iterated as a dict (yielding string
            # keys), crashing with "'str' object has no attribute 'get'".
            try:
                info = _ticker(sym).info
                rows.append({
                    "symbol": sym,
                    "name": info.get("longName", sym),
                    "sector": info.get("sector"),
                    "market_cap_B": _safe_float(
                        (info.get("marketCap") or 0) / 1e9, 2
                    ),
                    "pe_trailing": _safe_float(info.get("trailingPE")),
                    "price_to_book": _safe_float(info.get("priceToBook")),
                    "roe": _safe_float(info.get("returnOnEquity")),
                    "net_margin": _safe_float(info.get("profitMargins")),
                    "revenue_growth": _safe_float(info.get("revenueGrowth")),
                    "debt_to_equity": _safe_float(info.get("debtToEquity")),
                    "dividend_yield": _safe_float(info.get("dividendYield")),
                    "recommendation": info.get("recommendationKey"),
                })
            except Exception as exc:
                logger.warning("get_competitor_data: skipping %s after error: %s", sym, exc)
                continue

        return json.dumps(rows, default=str)
    except Exception as exc:
        logger.error("get_competitor_data failed for %s: %s", symbol, exc)
        return json.dumps(rows)   # still a list, even on an outer-level failure


@tool
def get_market_news(query: str, max_articles: int = 10) -> str:
    """
    Fetch recent market news articles relevant to a stock or topic.

    Sources: Yahoo Finance RSS (always) + News API (if key is configured).

    Args:
        query: Search term or ticker symbol, e.g. 'AAPL earnings' or 'IBM'.
        max_articles: Maximum articles to return (default 10, max 20).

    Returns:
        JSON array of {title, source, published, summary, url} objects.
    """
    max_articles = min(int(max_articles), 20)
    articles: list[dict] = []

    # ── Yahoo Finance RSS ────────────────────────────────────────────────────
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={query}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.get("title", ""),
                "source": "Yahoo Finance",
                "published": entry.get("published", ""),
                "summary": entry.get("summary", "")[:400],
                "url": entry.get("link", ""),
            })
    except Exception as exc:
        logger.warning("Yahoo Finance RSS failed: %s", exc)

    # ── News API (optional) ──────────────────────────────────────────────────
    if settings.NEWS_API_KEY and len(articles) < max_articles:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": max_articles - len(articles),
                    "language": "en",
                    "apiKey": settings.NEWS_API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            for art in resp.json().get("articles", []):
                articles.append({
                    "title": art.get("title", ""),
                    "source": art.get("source", {}).get("name", "News API"),
                    "published": art.get("publishedAt", ""),
                    "summary": (art.get("description") or "")[:400],
                    "url": art.get("url", ""),
                })
        except Exception as exc:
            logger.warning("News API failed: %s", exc)

    if not articles:
        return json.dumps({"message": "No news articles found", "query": query})
    return json.dumps(articles[:max_articles], default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Export list used by the agent
# ─────────────────────────────────────────────────────────────────────────────
ALL_FINANCIAL_TOOLS = [
    get_stock_info,
    get_historical_prices,
    get_financial_statements,
    calculate_financial_ratios,
    get_competitor_data,
    get_market_news,
]