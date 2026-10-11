"""
State definitions for the LangGraph Coding Agent.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class CodeChunk(TypedDict):
    """Represents a chunk of code retrieved from the codebase vector store."""
    file_path: str
    relative_path: str
    content: str
    start_line: int
    end_line: int
    score: Optional[float]


class PatchProposal(TypedDict, total=False):
    """Represents a targeted search-and-replace modification proposal."""
    file_path: str
    relative_path: str
    search_block: str
    replace_block: str
    explanation: str
    applied: bool
    error: Optional[str]
    backup_content: Optional[str]


class IterationRecord(TypedDict):
    """Record of a single attempt and human feedback."""
    iteration: int
    patch: Optional[PatchProposal]
    human_approved: bool
    is_resolved: bool
    human_feedback: str


class AgentState(TypedDict):
    """The central state maintained throughout the LangGraph workflow."""
    task: str                         # The plain English bug report or feature request
    codebase_dir: str                 # Root folder or target file path
    iteration: int                    # Current loop counter (1-based)
    max_iterations: int               # Maximum allowed iterations before stopping
    
    # Retrieval
    keywords: List[str]               # Keywords extracted by Groq LLM
    retrieved_chunks: List[CodeChunk] # Code snippets fetched from vector store
    
    # Patch & modification
    patch: Optional[PatchProposal]    # Latest proposed targeted patch
    diff_preview: str                 # Colored / unified diff of the changes
    
    # Human-in-the-loop review
    human_approved: bool              # True if human accepted the code changes
    is_resolved: bool                 # True if human confirmed the bug is resolved
    human_feedback: str               # Feedback/error description provided if not resolved
    
    # History & final status
    history: List[IterationRecord]    # History of all previous attempts and feedback
    status: str                       # e.g., 'IN_PROGRESS', 'RESOLVED', 'MAX_ITERATIONS_REACHED', 'ABORTED'
    final_message: str                # User-facing summary upon graph completion
