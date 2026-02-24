"""
Merge node: raccoglie tutti i ToolResult dalle esecuzioni parallele,
aggrega known_clients/known_products, gestisce pending_call e produce
la FinalResponse combinata.

Regola chiave per i carrelli:
- add_to_cart e remove_from_cart NON leggono il carrello internamente
  (evita race condition se paralleli per lo stesso cliente).
- Questo nodo legge il carrello UNA VOLTA per ogni client_id che ha subito
  una mutation, DOPO che tutti i branch hanno finito di scrivere.
"""

from src.graph.shared import AgentState, FinalResponse, WhatsAppSection, _db_manage_cart, _cart_text


def merge_node(state: AgentState) -> dict:
    results = state.get("tool_results") or []

    # Se il dispatcher ha già impostato final_answer (nessun tool call),
    # non c'è nulla da fare.
    if not results:
        return {}

    agent_code = state.get("agent_code", "")

    # Aggrega known_clients e known_products da tutti i branch
    merged_clients  = dict(state.get("known_clients") or {})
    merged_products = dict(state.get("known_products") or {})
    merged_memory   = list(state.get("session_memory") or [])
    for r in results:
        merged_clients.update(r.get("upd_clients") or {})
        merged_products.update(r.get("upd_products") or {})
        for fact in (r.get("upd_memory") or []):
            if fact not in merged_memory:
                merged_memory.append(fact)

    # Primo pending_call trovato (priorità: prima operazione che richiede input)
    pending_list = [
        r["pending_call"] for r in results
        if r.get("needs_input") and r.get("pending_call")
    ]
    new_pending = (
        pending_list[0] if len(pending_list) == 1
        else (pending_list if pending_list else None)
    )

    # Costruisce testo dai risultati che hanno testo esplicito
    # (add/remove hanno text="" e vengono esclusi — il carrello è aggiunto sotto)
    responses     = [FinalResponse(**r["response"]) for r in results]
    has_cart_view = any(r.get("cart_client_id") for r in results)
    non_empty     = [r for r in responses if r.text.strip()]
    lists         = [r for r in responses if r.sections or r.items]

    if len(pending_list) > 1:
        # BATCH: più disambiguazioni → testo unico, NO lista interattiva.
        # Evita round-trip multipli con l'LLM: l'utente risponde a tutto in un colpo.
        blocks = []
        needs_qty = False
        for r in results:
            if r.get("needs_input") and r.get("pending_call"):
                resp = FinalResponse(**r["response"])
                blocks.append(resp.text)
                if r["pending_call"].get("args", {}).get("quantity") is None:
                    needs_qty = True

        combined = "Ho bisogno di qualche precisazione:\n\n" + "\n\n".join(blocks)
        if needs_qty:
            combined += "\n\nSpecifica anche la quantità per ciascun prodotto."
        combined += "\n\nRispondi con le tue scelte."

        final = FinalResponse(text=combined, use_interactive_list=False)
    elif len(lists) > 1:
        # Fallback: più liste interattive ma un solo pending
        last_list = lists[-1]
        combined_text = "\n\n".join(r.text for r in non_empty)
        final = FinalResponse(
            text=combined_text,
            use_interactive_list=True,
            sections=last_list.sections or [],
            items=last_list.items or [],
        )
    elif len(responses) == 1 and not has_cart_view:
        final = responses[0]
    else:
        last_list     = lists[-1] if lists else None
        combined_text = "\n\n".join(r.text for r in non_empty)
        final = FinalResponse(
            text=combined_text,
            use_interactive_list=bool(last_list),
            items=last_list.items    if last_list else [],
            sections=last_list.sections if last_list else [],
        )

    # --- Cart view post-mutation ---
    # Raccoglie client_id unici che hanno avuto una mutation (add/remove)
    cart_client_ids: list = []
    for r in results:
        cid = r.get("cart_client_id")
        if cid and cid not in cart_client_ids:
            cart_client_ids.append(cid)

    if cart_client_ids:
        cart_parts = []
        for cid in cart_client_ids:
            cname   = (merged_clients.get(cid) or {}).get("ragione_sociale") or \
                      (merged_clients.get(cid) or {}).get("alias") or cid
            cart    = _db_manage_cart(agent_code, "view", cid) or []
            cart_text = _cart_text(cart if isinstance(cart, list) else [], cname)
            has_pending = new_pending is not None
            suffix  = "\n\nVuoi confermare e inviare l'ordine?" if isinstance(cart, list) and cart and not has_pending else ""
            cart_parts.append(f"{cart_text}{suffix}")

        cart_section = "\n\n".join(cart_parts)
        combined = f"{final.text}\n\n{cart_section}".strip() if final.text.strip() else cart_section
        # Preserva proprietà lista interattiva (pallino al secondo giro di disambiguazione)
        final = FinalResponse(
            text=combined,
            use_interactive_list=final.use_interactive_list,
            sections=final.sections,
            items=final.items,
            raw_data=final.raw_data,
            overflow=final.overflow,
        )

    print(f"   📤 Risposta: {final.text[:100]}")
    if final.use_interactive_list:
        n = sum(len(s.items) for s in final.sections) if final.sections else len(final.items)
        print(f"   📋 Lista: {n} elementi")
    if isinstance(new_pending, list):
        _pnames = ", ".join(pc["name"] for pc in new_pending)
        print(f"   ⏳ Pending ({len(new_pending)}): {_pnames}\n")
    else:
        print(f"   ⏳ Pending: {new_pending['name'] if new_pending else 'None'}\n")

    return {
        "final_answer":   final,
        "pending_call":   new_pending,
        "known_clients":  merged_clients,
        "known_products": merged_products,
        "session_memory": merged_memory,
    }
