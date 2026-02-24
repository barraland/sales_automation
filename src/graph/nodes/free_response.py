"""Nodo: risposta libera (benvenuto, out-of-scope, capacità)."""

from src.graph.shared import AgentState, FinalResponse


def free_response_node(state: AgentState) -> dict:
    tc   = state.get("current_tool_call") or {}
    args = tc.get("args") or {}
    text = args.get("text", "Come posso aiutarti?")

    return {
        "tool_results": [{
            "name":         "free_response",
            "response":     FinalResponse(text=text).model_dump(),
            "needs_input":  False,
            "pending_call": None,
            "upd_clients":    state.get("known_clients") or {},
            "upd_products":   state.get("known_products") or {},
            "cart_client_id": None,
        }]
    }
