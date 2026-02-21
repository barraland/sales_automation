"""
Sales Automation — Beverage Agent
Architettura fan-out con Send:

  dispatcher_node
      ↓ route_to_tools (Send per ogni tool call)
  [add_to_cart | remove_from_cart | clear_cart | view_cart |
   confirm_order | list_clients | search_products | list_orders | free_response]
      ↓ (tutti → merge)
  merge_node
      ↓
  END

I nodi tool girano in parallelo (LangGraph Send fan-out).
Il merge_node aggrega i risultati, gestisce pending_call e produce la FinalResponse.
"""

import os
import sqlite3

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# State e modelli condivisi
from src.graph.shared import AgentState, agent_name_map, _project_root

# agent_name_map è re-esportato: endpoint.py lo importa da qui
__all__ = ["create_graph", "agent_name_map"]

# Nodi
from src.graph.nodes.dispatcher     import dispatcher_node, route_to_tools
from src.graph.nodes.add_to_cart    import add_to_cart_node
from src.graph.nodes.remove_from_cart import remove_from_cart_node
from src.graph.nodes.clear_cart     import clear_cart_node
from src.graph.nodes.view_cart      import view_cart_node
from src.graph.nodes.confirm_order  import confirm_order_node
from src.graph.nodes.list_clients   import list_clients_node
from src.graph.nodes.search_products import search_products_node
from src.graph.nodes.list_orders    import list_orders_node
from src.graph.nodes.free_response  import free_response_node
from src.graph.nodes.merge          import merge_node


_TOOL_NODES = {
    "add_to_cart":      add_to_cart_node,
    "remove_from_cart": remove_from_cart_node,
    "clear_cart":       clear_cart_node,
    "view_cart":        view_cart_node,
    "confirm_order":    confirm_order_node,
    "list_clients":     list_clients_node,
    "search_products":  search_products_node,
    "list_orders":      list_orders_node,
    "free_response":    free_response_node,
}


def create_graph():
    wf = StateGraph(AgentState)

    # Nodi
    wf.add_node("dispatcher", dispatcher_node)
    for name, fn in _TOOL_NODES.items():
        wf.add_node(name, fn)
    wf.add_node("merge", merge_node)

    # Entry point
    wf.set_entry_point("dispatcher")

    # Fan-out: dispatcher → tool nodes (parallelo via Send) oppure → merge (nessun tool)
    wf.add_conditional_edges("dispatcher", route_to_tools)

    # Ogni tool node → merge
    for name in _TOOL_NODES:
        wf.add_edge(name, "merge")

    # Fine
    wf.add_edge("merge", END)

    _ckpt_path = os.path.join(_project_root, "checkpoints.db")
    conn = sqlite3.connect(_ckpt_path, check_same_thread=False)
    return wf.compile(checkpointer=SqliteSaver(conn))


if __name__ == "__main__":
    app = create_graph()
    for event in app.stream(
        {"question": "ciao", "chat_history": [], "agent_code": "AG001"},
        {"configurable": {"thread_id": "test"}},
    ):
        pass
