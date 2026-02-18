"""
IntelliAgent Demo
-----------------
Run this to see the full multi-agent pipeline in action.

Usage:
    python demo/run_demo.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()

console = Console()


def run_demo():
    from agents.orchestrator import run_pipeline

    query = "Analyze AAPL stock — is it a good short-term buy right now?"

    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", title="IntelliAgent"))
    console.print("\n[yellow]Running pipeline...[/yellow]\n")

    result = run_pipeline(query)

    console.print(Panel(result.get("research_output", ""), title="[green]Research Agent Output[/green]"))
    console.print(Panel(result.get("analysis_output", ""), title="[blue]Analysis Agent Output[/blue]"))
    console.print(Panel(
        f"[bold]Signal:[/bold] {result.get('signal')}\n"
        f"[bold]Confidence:[/bold] {result.get('confidence')}",
        title="[magenta]Signal[/magenta]"
    ))
    console.print(Panel(result.get("final_report", ""), title="[bold white]Final Report[/bold white]"))


if __name__ == "__main__":
    run_demo()
