"""
LangGraph nodes for the Groq Coding Agent.
Includes:
1. analyze_and_retrieve_node: LLM keyword extraction + Chroma vector search
2. propose_and_apply_patch_node: Groq LLM targeted patch generation + patcher execution
3. human_review_node: Interactive Human-In-The-Loop review, diff inspection, and approval
4. route_after_review: Conditional edge router for looping or terminating
"""

import json
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt

from .state import AgentState, CodeChunk, PatchProposal, IterationRecord
from .retriever import CodebaseRetriever
from .patcher import CodePatcher

console = Console()


def get_groq_llm() -> ChatGroq:
    """Initializes and returns the ChatGroq model instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
    
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0.1,
    )


# -------------------------------------------------------------------------
# Node 1: Analyze user request and retrieve relevant code chunks
# -------------------------------------------------------------------------

class QueryAnalysis(BaseModel):
    keywords: List[str] = Field(description="Relevant code keywords, function names, classes, or error symbols")
    search_query: str = Field(description="Focused semantic search query for the codebase")
    target_files_hint: List[str] = Field(default_factory=list, description="Likely target file names or patterns")


def analyze_and_retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Understands the human's plain English bug description,
    extracts relevant keywords using Groq LLM, and performs vector search.
    """
    iteration = state.get("iteration", 0)
    console.print(f"\n[bold cyan]─── [Iteration {iteration + 1}] Analyzing & Retrieving Code Context ───[/bold cyan]")

    llm = get_groq_llm()
    parser = JsonOutputParser(pydantic_object=QueryAnalysis)

    system_prompt = """You are an expert software engineer assistant.
Your job is to analyze a user's bug report or coding request, and optionally any feedback from a previous failed attempt.
Extract key code identifiers (function names, classes, variables, error types, file names) and formulate an optimal search query to find the relevant code chunk in the codebase vector store.

Respond strictly in valid JSON matching this schema:
{
  "keywords": ["symbol1", "symbol2"],
  "search_query": "focused semantic search query",
  "target_files_hint": ["optional_file.py"]
}"""

    user_prompt = f"""User Request / Bug Report:
{state['task']}

Current Iteration: {iteration + 1}
Previous Human Feedback (if any):
{state.get('human_feedback', 'None - this is the first attempt')}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    try:
        chain = prompt | llm | parser
        analysis: Dict[str, Any] = chain.invoke({})
        keywords = analysis.get("keywords", [])
        search_query = analysis.get("search_query", state["task"])
    except Exception as e:
        console.print(f"[yellow]Warning: Structured parsing failed ({e}). Using heuristic extraction.[/yellow]")
        keywords = [word for word in state["task"].split() if len(word) > 3][:5]
        search_query = state["task"]

    console.print(f"[dim]• Extracted Keywords: {keywords}[/dim]")
    console.print(f"[dim]• Vector Search Query: '{search_query}'[/dim]")

    # Vector search using CodebaseRetriever
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retriever = CodebaseRetriever(
        codebase_path=state["codebase_dir"],
        embedding_model_name=embedding_model,
    )
    retrieved_chunks = retriever.search(query=search_query, keywords=keywords, top_k=4)

    console.print(f"[green]✓ Found {len(retrieved_chunks)} relevant code chunk(s)[/green]")
    for i, chunk in enumerate(retrieved_chunks, 1):
        console.print(f"  [dim]{i}. {chunk['relative_path']} (lines {chunk['start_line']}-{chunk['end_line']})[/dim]")

    return {
        "keywords": keywords,
        "retrieved_chunks": retrieved_chunks,
    }


# -------------------------------------------------------------------------
# Node 2: Propose and apply targeted patch
# -------------------------------------------------------------------------

class PatchProposalSchema(BaseModel):
    file_path: str = Field(description="The exact relative or absolute file path to modify")
    search_block: str = Field(description="The EXACT lines from the existing file that need to be replaced. Must match existing code character-for-character.")
    replace_block: str = Field(description="The new replacement lines of code.")
    explanation: str = Field(description="Clear explanation of the fix and why it resolves the bug.")


def propose_and_apply_patch_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM receives the query + retrieved code context.
    The LLM changes the relevant part only (not the whole file or folder).
    Applies the patch and generates a unified diff.
    """
    console.print(f"\n[bold magenta]─── Generating Targeted Patch with Groq LLM ───[/bold magenta]")

    llm = get_groq_llm()
    parser = JsonOutputParser(pydantic_object=PatchProposalSchema)

    # Format the retrieved code context
    context_blocks = []
    for idx, chunk in enumerate(state["retrieved_chunks"], 1):
        context_blocks.append(
            f"--- Chunk #{idx} from File: {chunk['relative_path']} (Lines {chunk['start_line']}-{chunk['end_line']}) ---\n"
            f"{chunk['content']}\n"
        )
    code_context_str = "\n".join(context_blocks) if context_blocks else "No specific chunks retrieved."

    system_prompt = """You are a senior principal software engineer using Groq.
Your task is to fix the reported bug by modifying ONLY the relevant part of the code.

CRITICAL RULES:
1. Do NOT rewrite the entire file or folder.
2. Provide a 'search_block' containing the EXACT contiguous lines of code that currently exist in the file.
3. Provide a 'replace_block' containing your replacement lines.
4. Ensure indentation and syntax are 100% correct.
5. If human feedback from a previous attempt is provided, carefully heed it to avoid repeating mistakes.

Respond strictly in valid JSON matching this schema:
{
  "file_path": "relative/path/to/file.py",
  "search_block": "exact lines to replace",
  "replace_block": "new replacement lines",
  "explanation": "brief description of the fix"
}"""

    user_prompt = f"""User Bug Report / Task:
{state['task']}

Relevant Codebase Context:
{code_context_str}

Previous Human Feedback (if any):
{state.get('human_feedback', 'None')}

Provide the targeted fix in JSON format now:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    try:
        chain = prompt | llm | parser
        patch_json = chain.invoke({})
    except Exception as e:
        # Fallback raw invocation and extract JSON
        response = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
        raw_text = response.content
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end != 0:
            patch_json = json.loads(raw_text[start:end])
        else:
            raise RuntimeError(f"Failed to generate structured patch from LLM: {e}\nRaw output: {raw_text}")

    target_file = patch_json.get("file_path", "")
    search_block = patch_json.get("search_block", "")
    replace_block = patch_json.get("replace_block", "")
    explanation = patch_json.get("explanation", "Targeted code fix.")

    # Resolve target file path against codebase directory if relative
    if not os.path.isabs(target_file):
        abs_target_file = os.path.join(state["codebase_dir"], target_file)
    else:
        abs_target_file = target_file

    console.print(f"[dim]• Target File:[/dim] [cyan]{target_file}[/cyan]")
    console.print(f"[dim]• Explanation:[/dim] {explanation}")

    # Apply the targeted patch
    success, msg, backup_content, diff = CodePatcher.apply_patch(
        file_path=abs_target_file,
        search_block=search_block,
        replace_block=replace_block,
    )

    patch_proposal: PatchProposal = {
        "file_path": abs_target_file,
        "relative_path": target_file,
        "search_block": search_block,
        "replace_block": replace_block,
        "explanation": explanation,
        "applied": success,
        "error": None if success else msg,
        "backup_content": backup_content,
    }

    diff_str = diff if diff else (f"Failed to apply patch: {msg}" if not success else "No changes detected.")

    return {
        "patch": patch_proposal,
        "diff_preview": diff_str,
        "iteration": state.get("iteration", 0) + 1,
    }


# -------------------------------------------------------------------------
# Node 3: Human-in-the-Loop review and approval
# -------------------------------------------------------------------------

def human_review_node(state: AgentState) -> Dict[str, Any]:
    """
    Human-in-the-loop review.
    Shows the human the exact diff and explanation.
    Human indicates whether the code is approved and whether the bug is resolved.
    """
    console.print(f"\n[bold yellow]══════════ HUMAN IN THE LOOP REVIEW ══════════[/bold yellow]")

    patch = state.get("patch") or {}
    rel_path = patch.get("relative_path", "unknown")
    explanation = patch.get("explanation", "No explanation provided.")
    diff_text = state.get("diff_preview", "")
    applied = patch.get("applied", False)

    console.print(Panel(
        f"[bold]Modified File:[/bold] {rel_path}\n"
        f"[bold]Status:[/bold] {'[green]Applied[/green]' if applied else '[red]Failed to apply[/red]'}\n"
        f"[bold]Explanation:[/bold] {explanation}",
        title=f"Proposed Fix (Iteration {state['iteration']}/{state['max_iterations']})",
        border_style="cyan"
    ))

    if diff_text:
        console.print("\n[bold]Unified Diff:[/bold]")
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        console.print("[red]No diff available.[/red]")

    console.print("\n[bold]Options:[/bold]")
    console.print("  [bold green][1][/bold green] Approve & Mark Bug Resolved (Finish workflow)")
    console.print("  [bold yellow][2][/bold yellow] Not Resolved / Reject (Provide feedback for another attempt)")
    console.print("  [bold red][3][/bold red] Revert Changes & Abort")

    choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3"], default="1")

    history = list(state.get("history", []))

    if choice == "1":
        console.print("\n[bold green]✓ Fix approved by human! Bug marked as RESOLVED.[/bold green]")
        record: IterationRecord = {
            "iteration": state["iteration"],
            "patch": patch,
            "human_approved": True,
            "is_resolved": True,
            "human_feedback": "",
        }
        history.append(record)
        return {
            "human_approved": True,
            "is_resolved": True,
            "human_feedback": "",
            "history": history,
            "status": "RESOLVED",
            "final_message": f"Bug successfully resolved after {state['iteration']} iteration(s)!",
        }

    elif choice == "2":
        feedback = Prompt.ask(
            "\n[yellow]What is still broken or what should the agent change?[/yellow]",
            default="The bug is still not completely resolved. Please review the logic."
        )
        # Ask if user wants to keep the staged changes or revert to backup before retrying
        revert_choice = Prompt.ask("Revert this attempt before retrying? (y/n)", choices=["y", "n"], default="n")
        if revert_choice == "y" and patch.get("backup_content") and patch.get("file_path"):
            CodePatcher.revert_file(patch["file_path"], patch["backup_content"])
            console.print("[dim]Reverted file to pre-patch state.[/dim]")

        record: IterationRecord = {
            "iteration": state["iteration"],
            "patch": patch,
            "human_approved": False,
            "is_resolved": False,
            "human_feedback": feedback,
        }
        history.append(record)
        return {
            "human_approved": False,
            "is_resolved": False,
            "human_feedback": feedback,
            "history": history,
            "status": "IN_PROGRESS",
        }

    else:  # choice == "3"
        console.print("\n[bold red]Workflow aborted by human.[/bold red]")
        if patch.get("backup_content") and patch.get("file_path"):
            CodePatcher.revert_file(patch["file_path"], patch["backup_content"])
            console.print("[dim]Reverted file back to initial content.[/dim]")

        record: IterationRecord = {
            "iteration": state["iteration"],
            "patch": patch,
            "human_approved": False,
            "is_resolved": False,
            "human_feedback": "User aborted the session.",
        }
        history.append(record)
        return {
            "human_approved": False,
            "is_resolved": False,
            "human_feedback": "User aborted.",
            "history": history,
            "status": "ABORTED",
            "final_message": "Session was aborted by the human operator.",
        }


# -------------------------------------------------------------------------
# Conditional Edge Router
# -------------------------------------------------------------------------

def route_after_review(state: AgentState) -> str:
    """
    Decides whether to terminate the graph or loop back for another attempt.
    """
    if state.get("is_resolved", False):
        return "end_resolved"

    if state.get("status") == "ABORTED":
        return "end_aborted"

    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        console.print(f"\n[bold red]Reached maximum iterations ({state['max_iterations']}). Stopping loop.[/bold red]")
        return "end_max_iterations"

    # Not resolved yet -> repeat loop
    console.print("\n[cyan]↺ Looping back to LLM with human feedback...[/cyan]")
    return "retry_loop"
