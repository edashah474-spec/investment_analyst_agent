"""
main.py — CLI entry point for the Investment Analyst Agent
─────────────────────────────────────────────────────────────────────────────
Usage examples:

  # Structured analysis (fast, direct pipeline)
  python main.py analyse AAPL
  python main.py analyse TSLA --peers RIVN,GM,F --output report.json

  # Full agentic analysis (ReAct loop, more verbose)
  python main.py agent MSFT --peers GOOGL,AAPL,META

  # Ingest financial reports into RAG knowledge base
  python main.py ingest ./data/sample_reports/apple_10k.pdf

  # Launch the Streamlit dashboard
  python main.py dashboard
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from pipeline.analysis import build_full_analysis
from pipeline.agent import run_analysis
from rag.vector_store import ingest_documents

# Force UTF-8 on Windows so rich doesn't go through the legacy cp1252 console renderer
import sys as _sys, io as _io
if _sys.platform == "win32":
    _utf8_stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
else:
    _utf8_stdout = _sys.stdout

console = Console(highlight=False, file=_utf8_stdout)

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rich helpers
# ─────────────────────────────────────────────────────────────────────────────

RATING_COLOR = {
    "STRONG BUY": "bold green",
    "BUY": "green",
    "HOLD": "yellow",
    "UNDERWEIGHT": "dark_orange",
    "SELL": "bold red",
}

SENTIMENT_COLOR = {
    "BULLISH": "green",
    "NEUTRAL": "yellow",
    "BEARISH": "red",
}


def _print_analysis_report(result) -> None:
    """Pretty-print an AnalysisResult to the terminal."""
    # Header
    color = RATING_COLOR.get(result.rating, "white")
    console.print(
        Panel(
            f"[bold]{result.company_name}[/bold]  [{color}]{result.rating}[/{color}]\n"
            f"[dim]{result.symbol} • {result.sector}[/dim]",
            title="[bold blue]Investment Analyst Report[/bold blue]",
            border_style="blue",
        )
    )

    # Key Metrics table
    table = Table(title="Key Metrics", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Current Price", f"${result.current_price:,.2f}" if result.current_price else "N/A")
    table.add_row("Market Cap", f"${result.market_cap_b:.1f}B" if result.market_cap_b else "N/A")
    if result.target_price:
        table.add_row("AI Target Price (12m)", f"${result.target_price:,.2f}")
    if result.ratios:
        r = result.ratios
        table.add_row("P/E (Trailing)", str(r.get("pe_trailing", "N/A")))
        table.add_row("P/E (Forward)", str(r.get("pe_forward", "N/A")))
        table.add_row("Price / Book", str(r.get("price_to_book", "N/A")))
        table.add_row("ROE", f"{r.get('roe', 0)*100:.1f}%" if r.get("roe") else "N/A")
        table.add_row("Net Margin", f"{r.get('net_margin', 0)*100:.1f}%" if r.get("net_margin") else "N/A")
        table.add_row("Debt / Equity", str(r.get("debt_to_equity", "N/A")))
        table.add_row("Current Ratio", str(r.get("current_ratio", "N/A")))
    sent_color = SENTIMENT_COLOR.get(result.sentiment_label, "white")
    table.add_row(
        "Market Sentiment",
        f"[{sent_color}]{result.sentiment_label}[/{sent_color}] ({result.sentiment_score:+.2f})",
    )
    console.print(table)

    # Risk flags
    if result.risk_flags:
        console.print("\n[bold yellow]!! Risk Flags[/bold yellow]")
        for flag in result.risk_flags:
            console.print(f"  * {flag}")

    # Competitor comparison
    if result.competitor_table:
        ct = Table(title="Competitor Benchmarking", box=box.SIMPLE_HEAVY)
        cols = ["symbol", "market_cap_B", "pe_trailing", "roe", "net_margin",
                "revenue_growth", "debt_to_equity", "recommendation"]
        for col in cols:
            ct.add_column(col, justify="right" if col != "symbol" else "left")
        for row in result.competitor_table:
            ct.add_row(*[str(row.get(c, "")) for c in cols])
        console.print(ct)

    # News summary
    if result.news_summary:
        console.print("\n[bold]📰 Market Sentiment Summary[/bold]")
        console.print(result.news_summary)

    # AI Recommendation
    if result.recommendation:
        console.print(
            Panel(
                Markdown(result.recommendation),
                title="[bold green]AI-Generated Recommendation (IBM Granite)[/bold green]",
                border_style="green",
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI commands
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """
    Investment Analyst Agent — powered by LangChain + IBM Granite + Ollama (local, free)
    """


@cli.command()
@click.argument("symbol")
@click.option("--peers", default="", help="Comma-separated peer tickers, e.g. MSFT,GOOGL")
@click.option("--output", default=None, help="Save full result to a JSON file")
def analyse(symbol: str, peers: str, output: str | None) -> None:
    """Run the 6-step structured investment analysis (direct pipeline, fast)."""
    console.print(f"\n[bold blue]Starting analysis for [yellow]{symbol.upper()}[/yellow]…[/bold blue]\n")
    try:
        settings.validate()
    except EnvironmentError as exc:
        console.print(f"[bold red]Ollama not running:[/bold red] {exc}")
        sys.exit(1)

    result = build_full_analysis(symbol, competitors=peers)

    if result.error:
        console.print(f"[bold red]Analysis failed:[/bold red] {result.error}")
        sys.exit(1)

    _print_analysis_report(result)

    if output:
        data = {
            "symbol": result.symbol,
            "company_name": result.company_name,
            "sector": result.sector,
            "current_price": result.current_price,
            "market_cap_b": result.market_cap_b,
            "kpis": result.kpis,
            "ratios": result.ratios,
            "risk_flags": result.risk_flags,
            "sentiment_score": result.sentiment_score,
            "sentiment_label": result.sentiment_label,
            "news_summary": result.news_summary,
            "competitor_table": result.competitor_table,
            "recommendation": result.recommendation,
            "rating": result.rating,
            "target_price": result.target_price,
        }
        Path(output).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"\n[dim]Report saved to {output}[/dim]")


@cli.command()
@click.argument("symbol")
@click.option("--peers", default="", help="Comma-separated peer tickers")
@click.option("--focus", default="", help="Additional analysis focus or instructions")
def agent(symbol: str, peers: str, focus: str) -> None:
    """Run the full ReAct agentic analysis loop (verbose, tool-by-tool)."""
    console.print(f"\n[bold blue]Launching ReAct agent for [yellow]{symbol.upper()}[/yellow]…[/bold blue]\n")
    try:
        settings.validate()
    except EnvironmentError as exc:
        console.print(f"[bold red]Ollama not running:[/bold red] {exc}")
        sys.exit(1)

    result = run_analysis(symbol, competitors=peers, extra_context=focus)

    if result["error"]:
        console.print(f"[bold red]Agent error:[/bold red] {result['error']}")
    else:
        console.print(
            Panel(
                Markdown(result["report"]),
                title=f"[bold green]Agent Report — {result['symbol']}[/bold green]",
                border_style="green",
            )
        )
        if result["steps"]:
            console.print(f"\n[dim]Completed {len(result['steps'])} tool call(s).[/dim]")


@cli.command()
@click.argument("files", nargs=-1, required=True)
def ingest(files: tuple[str, ...]) -> None:
    """Ingest PDF or text financial reports into the RAG knowledge base."""
    try:
        settings.validate()
    except EnvironmentError as exc:
        console.print(f"[bold red]Ollama not running:[/bold red] {exc}")
        sys.exit(1)

    console.print(f"[bold blue]Ingesting {len(files)} file(s)…[/bold blue]")
    n = ingest_documents(list(files))
    console.print(f"[green]✓ {n} document chunks added to the knowledge base.[/green]")


@cli.command()
def dashboard() -> None:
    """Launch the Streamlit investment dashboard."""
    dashboard_path = Path(__file__).parent / "ui" / "dashboard.py"
    os.execlp("streamlit", "streamlit", "run", str(dashboard_path))


if __name__ == "__main__":
    cli()
