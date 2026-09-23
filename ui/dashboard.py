"""
ui/dashboard.py — Streamlit Investment Analyst Dashboard
─────────────────────────────────────────────────────────────────────────────
Launch with:
    streamlit run ui/dashboard.py
  or via CLI:
    python main.py dashboard

Features:
  • Ticker search + analysis trigger
  • Live KPI ledger
  • Interactive price chart (Plotly)
  • Financial ratio ledger
  • Competitor benchmarking table
  • Sentiment indicator
  • AI-generated recommendation (IBM Granite)
  • RAG document ingestion panel

Visual identity: private-research-note / boutique wealth-advisory theme —
ink background, gold hairlines, Fraunces + Inter type, tabular figures.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path when launched via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from pipeline.analysis import build_full_analysis, AnalysisResult
from rag.vector_store import ingest_documents
from tools.financial_data import get_historical_prices
import urllib.request


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investment Analyst Agent — IBM Granite",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Palette (single source of truth — used in CSS and in Plotly figures below)
# ─────────────────────────────────────────────────────────────────────────────
INK = "#0B0E14"
PANEL = "#131826"
LINE = "#232B3D"
GOLD = "#C6A15B"
GOLD_SOFT = "rgba(198,161,91,0.35)"
PARCHMENT = "#EDEAE2"
MUTED = "#8C93A6"
EMERALD = "#4C9A6A"
BRICK = "#B8614A"

RATING_COLOR = {
    "STRONG BUY": EMERALD, "BUY": EMERALD,
    "HOLD": GOLD,
    "UNDERWEIGHT": BRICK, "SELL": BRICK,
}
SENTIMENT_COLOR = {"BULLISH": EMERALD, "NEUTRAL": GOLD, "BEARISH": BRICK}


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class^="css"], [class*=" css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

.stApp {{
    background:
        radial-gradient(1100px 550px at 8% -10%, rgba(198,161,91,0.07), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(76,154,106,0.05), transparent 55%),
        {INK};
    color: {PARCHMENT};
}}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: #0E121B;
    border-right: 1px solid {LINE};
}}
[data-testid="stSidebar"] * {{ color: {PARCHMENT}; }}
[data-testid="stSidebar"] label {{
    color: {MUTED} !important; font-size: 0.75rem; letter-spacing: 0.03em;
}}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] textarea {{
    background: {INK} !important;
    color: {PARCHMENT} !important;
    border: 1px solid {LINE} !important;
    border-radius: 2px !important;
}}
.sidebar-title {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.5rem;
    color: {PARCHMENT}; letter-spacing: 0.01em; margin-bottom: 2px;
}}
.sidebar-sub {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 14px; }}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, {GOLD}, #8f7440);
    color: {INK}; border: none; border-radius: 2px;
    font-weight: 600; letter-spacing: 0.02em; padding: 0.55rem 1rem;
    transition: filter 0.15s ease, box-shadow 0.15s ease;
}}
.stButton > button:hover {{
    filter: brightness(1.12);
    box-shadow: 0 0 0 1px {GOLD_SOFT};
}}
.stButton > button:disabled {{
    background: {PANEL}; color: {MUTED};
}}

/* ── Alerts / expanders / dataframe chrome ──────────────────────────── */
[data-testid="stAlert"] {{
    background: {PANEL} !important; border: 1px solid {LINE} !important;
    border-radius: 2px !important; color: {PARCHMENT} !important;
}}
[data-testid="stExpander"] {{
    border: 1px solid {LINE} !important; border-radius: 2px !important;
    background: {PANEL} !important;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {LINE} !important; border-radius: 2px !important;
}}
[data-testid="stProgress"] > div > div > div > div {{
    background-color: {GOLD} !important;
}}

/* ── Header / letterhead ─────────────────────────────────────────────── */
.masthead {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.1rem;
    color: {PARCHMENT}; letter-spacing: 0.01em; margin-bottom: 2px;
}}
.masthead-rule {{
    height: 1px; background: linear-gradient(90deg, {GOLD}, transparent 70%);
    margin: 10px 0 18px 0;
}}
.masthead-sub {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 6px; }}

.company-name {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.2rem;
    color: {PARCHMENT}; margin: 0; line-height: 1.1;
}}
.company-meta {{ color: {MUTED}; font-size: 0.85rem; letter-spacing: 0.02em; }}

.seal {{
    width: 92px; height: 92px; border-radius: 50%;
    border: 1.5px solid var(--seal-color, {GOLD});
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; margin: 0 auto;
    background: radial-gradient(circle at 30% 30%, rgba(198,161,91,0.12), transparent 70%);
}}
.seal-rating {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 0.82rem;
    color: var(--seal-color, {GOLD}); text-align: center; line-height: 1.2;
    padding: 0 8px;
}}
.seal-target {{ text-align: center; color: {MUTED}; font-size: 0.75rem; margin-top: 8px; }}

.sentiment-line {{ font-size: 0.95rem; margin-bottom: 2px; }}
.sentiment-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
                   margin-right: 6px; background: var(--dot-color, {GOLD}); }}
.sentiment-score {{ color: {MUTED}; font-size: 0.78rem; }}

.timestamp-line {{ color: {MUTED}; font-size: 0.78rem; margin: 2px 0 18px 0; }}

/* ── Section headers ─────────────────────────────────────────────────── */
.section-head {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.15rem;
    color: {PARCHMENT}; margin: 6px 0 12px 0; display: flex; align-items: center;
}}
.section-head::before {{
    content: ""; display: inline-block; width: 7px; height: 7px;
    background: {GOLD}; margin-right: 10px; transform: rotate(45deg);
}}

/* ── Ledger (KPI row) ─────────────────────────────────────────────────── */
.ledger-row {{
    display: flex; flex-wrap: wrap;
    border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE};
    margin-bottom: 28px;
}}
.ledger-item {{
    flex: 1 1 160px; padding: 16px 20px; border-right: 1px solid {LINE};
}}
.ledger-item:last-child {{ border-right: none; }}
.ledger-label {{
    font-size: 0.72rem; color: {MUTED}; letter-spacing: 0.03em; margin-bottom: 6px;
}}
.ledger-value {{
    font-family: 'Fraunces', serif; font-size: 1.55rem; color: {PARCHMENT};
    font-variant-numeric: tabular-nums;
}}

/* ── Luxury tables ────────────────────────────────────────────────────── */
.lux-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
.lux-table th {{
    text-align: left; color: {MUTED}; font-weight: 500; font-size: 0.72rem;
    letter-spacing: 0.03em; padding: 8px 10px; border-bottom: 1px solid {GOLD_SOFT};
}}
.lux-table th.num, .lux-table td.num {{ text-align: right; }}
.lux-table td {{
    padding: 9px 10px; border-bottom: 1px solid {LINE}; color: {PARCHMENT};
    font-variant-numeric: tabular-nums;
}}
.lux-table td.label {{ color: {MUTED}; }}
.lux-table tr:last-child td {{ border-bottom: none; }}
.lux-table tr:hover td {{ background: rgba(198,161,91,0.045); }}
.lux-table td.primary {{ color: {GOLD}; font-weight: 600; }}

/* ── Risk flags ───────────────────────────────────────────────────────── */
.risk-row {{
    border-left: 2px solid {BRICK}; background: rgba(184,97,74,0.07);
    padding: 8px 14px; margin: 6px 0; font-size: 0.88rem; color: {PARCHMENT};
}}
.no-risk {{
    border-left: 2px solid {EMERALD}; background: rgba(76,154,106,0.07);
    padding: 8px 14px; font-size: 0.88rem; color: {PARCHMENT};
}}

/* ── Footer ───────────────────────────────────────────────────────────── */
.ibm-footer {{
    text-align: center; font-size: 0.75rem; color: {MUTED};
    border-top: 1px solid {LINE}; margin-top: 44px; padding-top: 14px;
    letter-spacing: 0.02em;
}}
.ibm-footer span {{ color: {GOLD}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _esc(val) -> str:
    return html.escape(str(val)) if val is not None else "—"


def _pct(val: float | None) -> str:
    """For fields yfinance returns as DECIMAL FRACTIONS (0.34 == 34%)."""
    return f"{val * 100:.1f}%" if val is not None else "N/A"


def _pct_already_scaled(val: float | None) -> str:
    """For fields yfinance already returns percentage-scale (0.78 == 0.78%)."""
    return f"{val:.2f}%" if val is not None else "N/A"


def _ratio_from_pct_scale(val: float | None) -> str:
    """Convert a percentage-scale value (29.12 == 29.12%) into a true 'x' ratio."""
    return f"{val / 100:.2f}x" if val is not None else "N/A"


def _fmt(val: float | None, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


def _section(text: str) -> None:
    st.markdown(f'<div class="section-head">{html.escape(text)}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration & RAG ingestion
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Investment Analyst</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">IBM Granite · Ollama (local) · LangChain</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Ollama health check
    try:
        urllib.request.urlopen(settings.OLLAMA_BASE_URL, timeout=2)
        ollama_ok = True
        st.success(f"Ollama running — `{settings.OLLAMA_MODEL}`")
    except Exception:
        ollama_ok = False
        st.warning(
            f"Ollama not reachable at `{settings.OLLAMA_BASE_URL}`\n\n"
            "Run: `ollama serve` then `ollama pull granite3.3:8b`"
        )

    st.subheader("Analysis Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL", max_chars=10).upper().strip()
    peers_input = st.text_input(
        "Competitor Tickers (comma-sep)",
        placeholder="MSFT,GOOGL,META",
    )
    period = st.selectbox("Price History Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True, disabled=not ollama_ok)

    st.divider()
    st.subheader("Ingest Financial Reports (RAG)")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT reports",
        accept_multiple_files=True,
        type=["pdf", "txt"],
    )
    if st.button("Ingest Documents", use_container_width=True):
        if uploaded_files:
            save_paths = []
            for uf in uploaded_files:
                dest = Path("./data/sample_reports") / uf.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(uf.getvalue())
                save_paths.append(str(dest))
            with st.spinner("Ingesting into RAG knowledge base…"):
                n = ingest_documents(save_paths)
            st.success(f"{n} chunks ingested from {len(save_paths)} file(s).")
        else:
            st.info("Upload files first.")

    st.divider()
    st.caption(f"Model: `{settings.OLLAMA_MODEL}` via Ollama")


# ─────────────────────────────────────────────────────────────────────────────
# Main area — masthead
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="masthead">Sequential Investment Analyst</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="masthead-sub">Data Gathering → KPI Extraction → Ratio Calculation → '
    'Competitor Benchmarking → Sentiment Analysis → AI Recommendation</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="masthead-rule"></div>', unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None


# ── Pipeline execution ───────────────────────────────────────────────────────
if run_btn:
    if not ollama_ok:
        st.error("Ollama is not running. Start it with: `ollama serve`")
    else:
        pipeline_steps = [
            "Step 1+2: Data Gathering & KPI Extraction",
            "Step 3: Financial Ratio Calculation",
            "Step 4: Competitor Benchmarking",
            "Step 5: Market Sentiment Analysis",
            "Step 6: AI Recommendation Generation",
        ]
        progress_bar = st.progress(0, text="Initialising pipeline…")
        status_area = st.empty()

        def _progress(i: int, msg: str) -> None:
            progress_bar.progress(i / len(pipeline_steps), text=msg)
            status_area.info(msg)

        _progress(0, pipeline_steps[0])
        result: AnalysisResult = build_full_analysis(ticker, competitors=peers_input)
        _progress(5, "Complete.")
        progress_bar.empty()
        status_area.empty()

        st.session_state.result = result
        if result.error:
            st.error(f"Analysis failed: {result.error}")


# ── Render results ───────────────────────────────────────────────────────────
result: AnalysisResult | None = st.session_state.result

if result and not result.error:

    # ── Header row ───────────────────────────────────────────────────────────
    rating_color = RATING_COLOR.get(result.rating, GOLD)
    sent_color = SENTIMENT_COLOR.get(result.sentiment_label, GOLD)

    col_title, col_sentiment, col_rating = st.columns([4, 2, 1.4])
    with col_title:
        st.markdown(f'<p class="company-name">{_esc(result.company_name)}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="company-meta">{_esc(result.symbol)} &nbsp;·&nbsp; {_esc(result.sector)}</div>',
            unsafe_allow_html=True,
        )
    with col_sentiment:
        st.markdown(
            f'<div class="sentiment-line" style="--dot-color:{sent_color}">'
            f'<span class="sentiment-dot"></span>{_esc(result.sentiment_label)} sentiment</div>'
            f'<div class="sentiment-score">score {result.sentiment_score:+.2f}</div>',
            unsafe_allow_html=True,
        )
    with col_rating:
        target_line = (
            f'<div class="seal-target">Target ${result.target_price:,.2f}</div>'
            if result.target_price else ""
        )
        st.markdown(
            f'<div class="seal" style="--seal-color:{rating_color}">'
            f'<div class="seal-rating">{_esc(result.rating)}</div></div>{target_line}',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="timestamp-line">Data as of {_esc(result.kpis.get("timestamp", "unknown"))}</div>',
        unsafe_allow_html=True,
    )

    # ── KPI ledger ───────────────────────────────────────────────────────────
    _section("Key Metrics")
    ledger_entries = [
        ("Current Price", _fmt(result.current_price, "$")),
        ("Market Cap", _fmt(result.market_cap_b, "$", "B", 1)),
        ("P/E (Trailing)", _fmt(result.ratios.get("pe_trailing"), "", "x")),
        ("ROE", _pct(result.ratios.get("roe"))),
        ("Net Margin", _pct(result.ratios.get("net_margin"))),
        ("Debt / Equity", _ratio_from_pct_scale(result.ratios.get("debt_to_equity"))),
    ]
    ledger_html = '<div class="ledger-row">' + "".join(
        f'<div class="ledger-item"><div class="ledger-label">{_esc(label)}</div>'
        f'<div class="ledger-value">{_esc(value)}</div></div>'
        for label, value in ledger_entries
    ) + "</div>"
    st.markdown(ledger_html, unsafe_allow_html=True)

    # ── Price chart ──────────────────────────────────────────────────────────
    _section("Price History")
    try:
        price_raw = get_historical_prices.invoke({"symbol": result.symbol, "period": period})
        price_data = json.loads(price_raw)
        if isinstance(price_data, list) and price_data:
            df_price = pd.DataFrame(price_data)
            if "Date" in df_price.columns:
                df_price["Date"] = pd.to_datetime(df_price["Date"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_price["Date"], y=df_price["Close"],
                    mode="lines", name="Close Price",
                    line=dict(color=GOLD, width=1.6),
                    fill="tozeroy", fillcolor="rgba(198,161,91,0.08)",
                ))
                fig.update_layout(
                    height=340,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title=None, yaxis_title="Price (USD)",
                    hovermode="x unified",
                    plot_bgcolor=PANEL,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color=PARCHMENT, size=12),
                    xaxis=dict(gridcolor=LINE, showline=False, zeroline=False),
                    yaxis=dict(gridcolor=LINE, showline=False, zeroline=False),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price data not available.")
    except Exception as exc:
        st.warning(f"Price chart error: {exc}")

    # ── Ratio ledger + risk flags ─────────────────────────────────────────────
    col_ratios, col_risks = st.columns([3, 2])

    with col_ratios:
        _section("Financial Ratios")
        ratio_items = {
            "P/E (Trailing)": result.ratios.get("pe_trailing"),
            "P/E (Forward)": result.ratios.get("pe_forward"),
            "Price / Book": result.ratios.get("price_to_book"),
            "EV / EBITDA": result.ratios.get("ev_to_ebitda"),
            "PEG Ratio": result.ratios.get("peg_ratio"),
            "ROE": result.ratios.get("roe"),
            "ROA": result.ratios.get("roa"),
            "Gross Margin": result.ratios.get("gross_margin"),
            "Operating Margin": result.ratios.get("operating_margin"),
            "Net Margin": result.ratios.get("net_margin"),
            "Current Ratio": result.ratios.get("current_ratio"),
            "Quick Ratio": result.ratios.get("quick_ratio"),
            "Debt / Equity": result.ratios.get("debt_to_equity"),
            "Dividend Yield": result.ratios.get("dividend_yield"),
            "FCF Yield": result.ratios.get("free_cashflow_yield"),
        }
        # Fields yfinance returns as DECIMAL FRACTIONS → need ×100 for %.
        pct_fields = {"ROE", "ROA", "Gross Margin", "Operating Margin",
                      "Net Margin", "FCF Yield"}

        rows_html = []
        for name, val in ratio_items.items():
            if val is None:
                continue
            if name in pct_fields:
                display = f"{val * 100:.1f}%"
            elif name == "Dividend Yield":
                display = f"{val:.2f}%"          # already percentage-scale
            elif name == "Debt / Equity":
                display = f"{val / 100:.2f}x"    # already percentage-scale
            else:
                display = f"{val:.2f}"
            rows_html.append(
                f'<tr><td class="label">{_esc(name)}</td><td class="num">{_esc(display)}</td></tr>'
            )
        if rows_html:
            st.markdown(
                '<table class="lux-table"><tbody>' + "".join(rows_html) + '</tbody></table>',
                unsafe_allow_html=True,
            )

    with col_risks:
        _section("Risk Indicators")
        if result.risk_flags:
            for flag in result.risk_flags:
                st.markdown(f'<div class="risk-row">{_esc(flag)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-risk">No major risk flags detected.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _section("Sentiment Summary")
        st.markdown(
            f'<div class="sentiment-line" style="--dot-color:{sent_color}">'
            f'<span class="sentiment-dot"></span>{_esc(result.sentiment_label)}</div>',
            unsafe_allow_html=True,
        )
        if result.news_summary:
            st.write(result.news_summary)

    # ── Competitor benchmarking ───────────────────────────────────────────────
    # Guard against a non-list competitor_table (e.g. an error object from an
    # older tools/financial_data.py) — iterating a dict yields its string
    # keys, which previously crashed the `r.get(...)` calls below.
    if isinstance(result.competitor_table, list) and result.competitor_table:
        _section("Competitor Benchmarking")

        pe_data = [(r.get("symbol"), r.get("pe_trailing")) for r in result.competitor_table
                   if r.get("pe_trailing")]
        if pe_data:
            fig_comp = go.Figure(go.Bar(
                x=[d[0] for d in pe_data],
                y=[d[1] for d in pe_data],
                marker_color=[GOLD if d[0] == result.symbol else "#3A4258" for d in pe_data],
                text=[f"{d[1]:.1f}x" for d in pe_data],
                textposition="outside",
                textfont=dict(color=PARCHMENT, family="Inter"),
            ))
            fig_comp.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=20, b=0),
                plot_bgcolor=PANEL, paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color=PARCHMENT, size=12),
                yaxis=dict(title="P/E Ratio", gridcolor=LINE, zeroline=False),
                xaxis=dict(gridcolor=LINE, zeroline=False),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        headers = ["Symbol", "Name", "Mkt Cap ($B)", "P/E", "ROE", "Net Margin",
                   "Rev Growth", "Debt/Equity", "Recommendation"]
        header_html = "".join(
            f'<th class="num">{h}</th>' if h not in ("Symbol", "Name", "Recommendation")
            else f"<th>{h}</th>"
            for h in headers
        )
        comp_rows = []
        for r in result.competitor_table:
            is_primary = r.get("symbol") == result.symbol
            sym_cls = "primary" if is_primary else ""
            comp_rows.append(
                "<tr>"
                f'<td class="{sym_cls}">{_esc(r.get("symbol"))}</td>'
                f'<td>{_esc(r.get("name"))}</td>'
                f'<td class="num">{_fmt(r.get("market_cap_B"), decimals=1)}</td>'
                f'<td class="num">{_fmt(r.get("pe_trailing"), suffix="x")}</td>'
                f'<td class="num">{_pct(r.get("roe"))}</td>'
                f'<td class="num">{_pct(r.get("net_margin"))}</td>'
                f'<td class="num">{_pct(r.get("revenue_growth"))}</td>'
                f'<td class="num">{_ratio_from_pct_scale(r.get("debt_to_equity"))}</td>'
                f'<td>{_esc(r.get("recommendation") or "—")}</td>'
                "</tr>"
            )
        st.markdown(
            f'<table class="lux-table"><thead><tr>{header_html}</tr></thead>'
            f'<tbody>{"".join(comp_rows)}</tbody></table>',
            unsafe_allow_html=True,
        )

    # ── AI Recommendation ────────────────────────────────────────────────────
    _section("AI-Generated Investment Recommendation")
    st.caption(f"Generated by IBM Granite (`{settings.OLLAMA_MODEL}`) running locally via Ollama")
    if result.recommendation:
        with st.expander("View Full Recommendation Report", expanded=True):
            st.markdown(result.recommendation)

    # ── Pipeline trace ───────────────────────────────────────────────────────
    with st.expander("Analysis Pipeline Trace"):
        steps = [
            ("01", "Data Gathering & KPI Extraction", "get_stock_info, get_financial_statements, search_reports"),
            ("02", "Financial Ratio Calculation", "calculate_financial_ratios"),
            ("03", "Competitor Benchmarking", "get_competitor_data"),
            ("04", "Market Sentiment Analysis", "get_market_news → IBM Granite sentiment"),
            ("05", "Investment Recommendation", f"IBM Granite ({settings.OLLAMA_MODEL}) via Ollama"),
        ]
        for step_id, step_name, tools_used in steps:
            st.markdown(
                f'<div style="margin:6px 0;"><span style="color:{GOLD};font-family:Fraunces,serif;'
                f'font-weight:600;">{step_id}</span>&nbsp;&nbsp;<strong>{_esc(step_name)}</strong>'
                f'&nbsp;— <span style="color:{MUTED};">{_esc(tools_used)}</span></div>',
                unsafe_allow_html=True,
            )

elif result and result.error:
    st.error(f"Analysis error: {result.error}")
else:
    # Welcome state
    st.info("Enter a ticker symbol in the sidebar and click Run Analysis to start.")
    with st.expander("How the pipeline works"):
        st.markdown(f"""
| Step | Description | Tools Used |
|------|-------------|-----------|
| 1+2 | **Data Gathering & KPI Extraction** | `get_stock_info`, `get_financial_statements`, `search_reports` (RAG) |
| 3 | **Financial Ratio Calculation** | `calculate_financial_ratios` (P/E, ROE, D/E, margins …) |
| 4 | **Competitor Benchmarking** | `get_competitor_data` (side-by-side peer comparison) |
| 5 | **Market Sentiment Analysis** | `get_market_news` → IBM Granite NLP |
| 6 | **Investment Recommendation** | IBM Granite `{settings.OLLAMA_MODEL}` via Ollama (local) |
        """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="ibm-footer">Made with IBM Bob <span>·</span> IBM Granite (Ollama) '
    f'<span>·</span> LangChain <span>·</span> 100% Free &amp; Local</div>',
    unsafe_allow_html=True,
)