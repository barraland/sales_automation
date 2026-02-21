"""Nodo: cerca prodotti nel catalogo (informativo)."""

from src.graph.shared import AgentState, impl_search_products


def search_products_node(state: AgentState) -> dict:
    tc   = state.get("current_tool_call") or {}
    args = tc.get("args") or {}

    resp, needs_input, pending_call, upd_c, upd_p, cart_client_id = impl_search_products(
        args.get("query", ""),
        args.get("filters"),
        state.get("agent_code", ""),
        state.get("known_clients") or {},
        state.get("known_products") or {},
    )

    return {
        "tool_results": [{
            "name":         "search_products",
            "response":     resp.dict(),
            "needs_input":  needs_input,
            "pending_call": pending_call,
            "upd_clients":    upd_c,
            "upd_products":   upd_p,
            "cart_client_id": cart_client_id,
        }]
    }
