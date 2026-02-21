"""Nodo: mostra gli ordini passati dell'agente."""

from src.graph.shared import AgentState, impl_list_orders


def list_orders_node(state: AgentState) -> dict:
    tc   = state.get("current_tool_call") or {}
    args = tc.get("args") or {}

    resp, needs_input, pending_call, upd_c, upd_p, cart_client_id = impl_list_orders(
        args.get("client_ref"),
        state.get("agent_code", ""),
        state.get("known_clients") or {},
        state.get("known_products") or {},
    )

    return {
        "tool_results": [{
            "name":         "list_orders",
            "response":     resp.dict(),
            "needs_input":  needs_input,
            "pending_call": pending_call,
            "upd_clients":    upd_c,
            "upd_products":   upd_p,
            "cart_client_id": cart_client_id,
        }]
    }
