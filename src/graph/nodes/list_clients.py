"""Nodo: mostra la lista clienti dell'agente."""

from src.graph.shared import AgentState, impl_list_clients


def list_clients_node(state: AgentState) -> dict:
    tc   = state.get("current_tool_call") or {}
    args = tc.get("args") or {}

    resp, needs_input, pending_call, upd_c, upd_p, cart_client_id = impl_list_clients(
        args.get("query"),
        args.get("city_filter"),
        state.get("agent_code", ""),
        state.get("known_clients") or {},
        state.get("known_products") or {},
    )

    return {
        "tool_results": [{
            "name":         "list_clients",
            "response":     resp.dict(),
            "needs_input":  needs_input,
            "pending_call": pending_call,
            "upd_clients":    upd_c,
            "upd_products":   upd_p,
            "cart_client_id": cart_client_id,
        }]
    }
