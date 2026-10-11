"""
Graph definition and compilation using LangGraph.
Constructs the iterative self-healing state graph with Human-in-the-Loop.
"""

from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    analyze_and_retrieve_node,
    propose_and_apply_patch_node,
    human_review_node,
    route_after_review,
)


def build_coding_agent_graph():
    """
    Builds and compiles the LangGraph StateGraph.

    Flow:
        [START]
           │
           ▼
    [analyze_and_retrieve]  ◄──────────────────────────────┐
           │                                               │
           ▼                                               │
    [propose_and_apply_patch]                              │
           │                                               │
           ▼                                               │
    [human_review] ─── (retry_loop: not resolved) ─────────┘
           │
           ├─── (end_resolved) ───────► [END]
           ├─── (end_aborted) ────────► [END]
           └─── (end_max_iterations) ─► [END]
    """
    workflow = StateGraph(AgentState)

    # 1. Add nodes
    workflow.add_node("analyze_and_retrieve", analyze_and_retrieve_node)
    workflow.add_node("propose_and_apply_patch", propose_and_apply_patch_node)
    workflow.add_node("human_review", human_review_node)

    # 2. Set entry point
    workflow.set_entry_point("analyze_and_retrieve")

    # 3. Add direct transitions
    workflow.add_edge("analyze_and_retrieve", "propose_and_apply_patch")
    workflow.add_edge("propose_and_apply_patch", "human_review")

    # 4. Add conditional routing based on human review
    workflow.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "retry_loop": "analyze_and_retrieve",
            "end_resolved": END,
            "end_aborted": END,
            "end_max_iterations": END,
        },
    )

    # 5. Compile the graph
    app = workflow.compile()
    return app
