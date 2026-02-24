"""Nodo: rimuove un prodotto dal carrello di un cliente."""

from src.graph.shared import AgentState, impl_remove_from_cart


def remove_from_cart_node(state: AgentState) -> dict:
    tc   = state.get("current_tool_call") or {}
    args = tc.get("args") or {}

    print(f"   ▶ [REMOVE_FROM_CART] client={args.get('client_ref')}, product={args.get('product_ref')}, "
          f"city={args.get('city_hint')}")

    resp, needs_input, pending_call, upd_c, upd_p, cart_client_id = impl_remove_from_cart(
        args.get("client_ref", ""),
        args.get("product_ref", ""),
        state.get("agent_code", ""),
        state.get("known_clients") or {},
        state.get("known_products") or {},
        city_hint=args.get("city_hint"),
    )

    return {
        "tool_results": [{
            "name":           "remove_from_cart",
            "response":       resp.model_dump(),
            "needs_input":    needs_input,
            "pending_call":   pending_call,
            "upd_clients":    upd_c,
            "upd_products":   upd_p,
            "cart_client_id": cart_client_id,
        }]
    }
