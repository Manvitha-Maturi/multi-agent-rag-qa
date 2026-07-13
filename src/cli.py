"""
Interactive CLI for the multi-agent lithium-sulfur RAG QA system.

Usage:
    python -m src.cli            # normal mode
    python -m src.cli --verbose  # also show the per-stage agent trace

Every value shown is produced by the pipeline itself (route, sources,
grounded/refused verdict, retry count, wall-clock time). Nothing is
synthesized for display.
"""
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich import box

from src.orchestrator import run_pipeline

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"
console = Console()

MODEL = "Groq / Llama-3.3-70B"
CORPUS = "11 lithium-sulfur battery PDFs"


def banner():
    body = Text()
    body.append("MULTI-AGENT TECHNICAL QA SYSTEM", style="bold cyan")
    body.append("  v1.0\n", style="dim")
    body.append("Corpus:  ", style="bold");  body.append(f"{CORPUS}\n")
    body.append("Index:   ", style="bold");  body.append("FAISS (local) · all-mpnet-base-v2\n")
    body.append("Model:   ", style="bold");  body.append(f"{MODEL}\n")
    body.append("Agents:  ", style="bold")
    body.append("router → retrieval → answerer → critic", style="cyan")
    console.print(Panel(body, box=box.ROUNDED, border_style="cyan", padding=(1, 2)))
    console.print("Ask a question, or type [bold]exit[/bold] to quit.\n", style="dim")


def verdict(result):
    """Return (label, style) from real pipeline fields — no invented state."""
    if getattr(result, "route", "") == "out_of_scope":
        return "OUT OF SCOPE — outside the corpus, not answered", "yellow"
    if getattr(result, "refused", False):
        return "REFUSED — retrieved context did not support an answer", "yellow"
    if getattr(result, "verified", False):
        return "GROUNDED — answer verified against retrieved sources", "green"
    return "UNVERIFIED — critic could not fully ground this answer", "orange1"


def clean_answer(result):
    """Strip the internal refusal sentinel for display; keep its explanation."""
    ans = (getattr(result, "answer", "") or "").strip()
    if getattr(result, "refused", False) and ans.upper().startswith(REFUSAL_SENTINEL):
        parts = ans.split("\n", 1)
        return parts[1].strip() if len(parts) > 1 else "Retrieved context did not support an answer."
    return ans


def sources_table(sources):
    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Retrieved source document", style="cyan")
    for i, s in enumerate(sources, 1):
        table.add_row(str(i), s)
    return table


def show(result, elapsed):
    label, style = verdict(result)
    route = getattr(result, "route", "?")

    console.print(f"\n[bold]Route:[/bold] {route}   [{style}]{label}[/{style}]")

    sources = getattr(result, "sources", [])
    if sources:
        console.print(sources_table(sources))

    answer = clean_answer(result)
    border = "green" if getattr(result, "verified", False) and not getattr(result, "refused", False) else style
    console.print(Panel(Markdown(answer), title="Answer", title_align="left",
                        border_style=border, box=box.ROUNDED, padding=(1, 2)))

    if VERBOSE:
        for line in getattr(result, "trace", []):
            console.print(f"  [dim]trace:[/dim] {line}")
        console.print(f"  [dim]trace:[/dim] retries used: {getattr(result, 'retries', 0)}")

    grounded = "yes" if getattr(result, "verified", False) and not getattr(result, "refused", False) else "no"
    console.print(
        f"[dim]✓ completed in {elapsed:.2f}s | route: {route} | "
        f"grounded: {grounded} | retries: {getattr(result, 'retries', 0)}[/dim]\n"
    )
    console.rule(style="dim")


def main():
    console.clear()
    banner()
    if VERBOSE:
        console.print("[dim](verbose: per-stage agent trace enabled)[/dim]\n")

    while True:
        try:
            question = console.input("[bold cyan]Q>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Session ended.[/dim]")
            break

        start = time.perf_counter()
        try:
            with console.status("[bold cyan]Running multi-agent pipeline…", spinner="dots"):
                result = run_pipeline(question)
        except Exception as e:
            console.print(Panel(f"{type(e).__name__}: {e}", title="Pipeline error",
                                border_style="red", box=box.ROUNDED))
            continue
        elapsed = time.perf_counter() - start
        show(result, elapsed)


if __name__ == "__main__":
    main()