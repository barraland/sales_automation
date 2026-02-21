"""Nodo: aggiunge un prodotto al carrello di un cliente."""

from src.graph.shared import AgentState, impl_add_to_cart


def add_to_cart_node(state: AgentState) -> dict:
    tc   = state.get("current_tool_call") or {}
    args = tc.get("args") or {}

    resp, needs_input, pending_call, upd_c, upd_p, cart_client_id = impl_add_to_cart(
        args.get("client_ref", ""),
        args.get("product_ref", ""),
        args.get("quantity"),
        state.get("agent_code", ""),
        state.get("known_clients") or {},
        state.get("known_products") or {},
    )

    return {
        "tool_results": [{
            "name":           "add_to_cart",
            "response":       resp.dict(),
            "needs_input":    needs_input,
            "pending_call":   pending_call,
            "upd_clients":    upd_c,
            "upd_products":   upd_p,
            "cart_client_id": cart_client_id,  # non-None → merge farà view_cart
        }]
    }
