"""Nodo: query in linguaggio naturale su SQLite (prodotti, clienti, ordini)."""

from src.graph.shared import AgentState, impl_query_database


def query_database_node(state: AgentState) -> dict:
    tc   = state.get("current_tool_call") or {}
    args = tc.get("args") or {}

    resp, needs_input, pending_call, upd_c, upd_p, cart_client_id = impl_query_database(
        args.get("question", ""),
        state.get("known_clients") or {},
        state.get("known_products") or {},
        agent_code=state.get("agent_code", ""),
        client_hint=args.get("client_hint"),
        product_hint=args.get("product_hint"),
    )

    return {
        "tool_results": [{
            "name":           "query_database",
            "response":       resp.dict(),
            "needs_input":    needs_input,
            "pending_call":   pending_call,
            "upd_clients":    upd_c,
            "upd_products":   upd_p,
            "cart_client_id": cart_client_id,
        }]
    }
