"""
Main entry point for the Groq LangGraph Coding Agent.
Handles CLI arguments, interactive prompts, codebase indexing, and execution.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

# Load environment variables from .env file
load_dotenv()

from agent.graph import build_coding_agent_graph
from agent.retriever import CodebaseRetriever
from agent.state import AgentState

console = Console()


def print_banner():
    banner_text = """[bold cyan]╔══════════════════════════════════════════════════════════════╗
║             GROQ + LANGGRAPH CODING AGENT                   ║
║   Iterative Bug Fixing with Vector Search & Human-In-The-Loop║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]"""
    console.print(banner_text)


def check_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "gsk_your_groq_api_key_here":
        console.print("\n[bold red]Error: GROQ_API_KEY is not set![/bold red]")
        console.print("Please get an API key from [link=https://console.groq.com/keys]https://console.groq.com/keys[/link]")
        entered_key = Prompt.ask("Enter your Groq API Key now (or press Enter to exit)")
        if not entered_key.strip():
            sys.exit(1)
        os.environ["GROQ_API_KEY"] = entered_key.strip()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Autonomous Coding Agent powered by Groq and LangGraph."
    )
    parser.add_argument(
        "--path", "-p",
        type=str,
        help="Path to the codebase folder or target code file."
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Plain English description of the bug or task."
    )
    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=int(os.getenv("MAX_ITERATIONS", "5")),
        help="Maximum self-healing refinement loops (default: 5)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        help="Groq LLM model name (default: llama-3.3-70b-versatile)."
    )
    return parser.parse_args()


def main():
    print_banner()
    check_api_key()
    args = parse_arguments()

    # Determine codebase path
    codebase_path_str = args.path
    if not codebase_path_str:
        default_path = "./sample_codebase" if Path("./sample_codebase").exists() else "."
        codebase_path_str = Prompt.ask(
            "\n[bold]Enter codebase folder or file path[/bold]",
            default=default_path
        )

    codebase_path = Path(codebase_path_str).resolve()
    if not codebase_path.exists():
        console.print(f"[bold red]Error: Path '{codebase_path}' does not exist![/bold red]")
        sys.exit(1)

    # Determine user task
    task = args.task
    if not task:
        task = Prompt.ask(
            "\n[bold]Describe the bug or feature request in plain English[/bold]"
        )

    if not task.strip():
        console.print("[bold red]Error: Task description cannot be empty.[/bold red]")
        sys.exit(1)

    # Index codebase into Vector Store
    console.print(f"\n[bold blue]Indexing codebase at:[/bold blue] {codebase_path}")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retriever = CodebaseRetriever(
        codebase_path=str(codebase_path),
        embedding_model_name=embedding_model,
    )
    total_chunks = retriever.index_codebase()
    console.print(f"[green]✓ Successfully indexed {total_chunks} code chunk(s) into vector store.[/green]")

    # Initial state for LangGraph
    initial_state: AgentState = {
        "task": task,
        "codebase_dir": str(codebase_path),
        "iteration": 0,
        "max_iterations": args.max_iterations,
        "keywords": [],
        "retrieved_chunks": [],
        "patch": None,
        "diff_preview": "",
        "human_approved": False,
        "is_resolved": False,
        "human_feedback": "",
        "history": [],
        "status": "IN_PROGRESS",
        "final_message": "",
    }

    # Build and execute the graph
    console.print("\n[bold green]Starting LangGraph workflow...[/bold green]")
    app = build_coding_agent_graph()

    final_state = app.invoke(initial_state)

    # Display final execution summary
    console.print("\n" + "=" * 60)
    status = final_state.get("status", "COMPLETED")
    resolved = final_state.get("is_resolved", False)

    if resolved:
        console.print(Panel(
            f"[bold green]Bug Successfully Resolved![/bold green]\n"
            f"Total Iterations: {final_state.get('iteration', 1)}\n"
            f"Modified File: {final_state.get('patch', {}).get('relative_path', 'N/A')}\n\n"
            f"[dim]{final_state.get('final_message', '')}[/dim]",
            title="Execution Result",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]Status: {status}[/bold yellow]\n"
            f"Resolved: {'Yes' if resolved else 'No'}\n"
            f"Total Iterations: {final_state.get('iteration', 1)}\n"
            f"Message: {final_state.get('final_message', 'Process ended.')}",
            title="Execution Result",
            border_style="yellow"
        ))

    # Print iteration history table
    history = final_state.get("history", [])
    if history:
        table = Table(title="Iteration History Summary")
        table.add_column("Iteration", style="cyan", justify="center")
        table.add_column("Approved", style="magenta", justify="center")
        table.add_column("Resolved", style="green", justify="center")
        table.add_column("Human Feedback", style="white")

        for record in history:
            table.add_row(
                str(record.get("iteration")),
                "✓" if record.get("human_approved") else "✗",
                "✓" if record.get("is_resolved") else "✗",
                record.get("human_feedback") or "N/A"
            )
        console.print(table)


if __name__ == "__main__":
    main()
