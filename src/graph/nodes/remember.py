"""Nodo: memorizza un fatto/alias/preferenza dell'utente nella session_memory."""

from src.graph.shared import AgentState


def remember_node(state: AgentState) -> dict:
    tc   = state.get("current_tool_call") or {}
    args = tc.get("args") or {}
    fact = args.get("fact", "").strip()

    if not fact:
        return {
            "tool_results": [{
                "name":           "remember",
                "response":       {"text": ""},
                "needs_input":    False,
                "pending_call":   None,
                "upd_clients":    {},
                "upd_products":   {},
                "cart_client_id": None,
                "upd_memory":     [],
            }]
        }

    print(f"   📝 [REMEMBER] {fact}")

    return {
        "tool_results": [{
            "name":           "remember",
            "response":       {"text": f"📝 Mi ricorderò che {fact}"},
            "needs_input":    False,
            "pending_call":   None,
            "upd_clients":    {},
            "upd_products":   {},
            "cart_client_id": None,
            "upd_memory":     [fact],
        }]
    }
