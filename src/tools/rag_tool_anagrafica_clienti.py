"""Ricerca clienti tramite Weaviate hybrid search (BM25 + vector)."""

import os
import weaviate.classes.query as wq

from src.tools.weaviate_client import get_collection


def _to_result(obj) -> dict:
    """Converte un oggetto Weaviate nel formato dict standard."""
    props = obj.properties
    result = {
        "client_id":        props.get("client_id", ""),
        "ragione_sociale":  props.get("ragione_sociale", ""),
        "alias":            props.get("alias", ""),
        "indirizzo":        props.get("indirizzo", ""),
        "citta":            props.get("citta", ""),
        "full_address":     f"{props.get('indirizzo', '')}, {props.get('citta', '')}".strip(", "),
    }
    # Score disponibile solo con hybrid search
    if obj.metadata and obj.metadata.score is not None:
        result["score"] = round(obj.metadata.score, 4)
    return result


def search_client_smart(
    agent_code: str,
    query_text: str = None,
    client_id: str = None,
    city_filter: str = None,
) -> list[dict]:
    """
    Ricerca clienti con Weaviate hybrid search.

    Firma e return identici alla versione FTS5 precedente:
    list[dict] con chiavi: client_id, ragione_sociale, alias, indirizzo, citta, full_address.
    """
    if not agent_code:
        return []

    try:
        collection = get_collection()
    except Exception as e:
        print(f"⚠️ [WEAVIATE] Connessione fallita: {e}")
        return []

    agent_filter = wq.Filter.by_property("agent_id").equal(agent_code)

    _meta = wq.MetadataQuery(score=True)

    try:
        # CASO A: Selezione diretta tramite client_id (es. da bottone WhatsApp)
        if client_id:
            id_filter = (
                wq.Filter.by_property("client_id").equal(client_id) &
                agent_filter
            )
            resp = collection.query.fetch_objects(filters=id_filter, limit=1)
            return [_to_result(o) for o in resp.objects]

        # CASO B1: Solo filtro città (es. "clienti di Milano")
        if city_filter and not query_text:
            resp = collection.query.hybrid(
                query=city_filter,
                filters=agent_filter,
                alpha=0.3,  # peso maggiore a BM25 per match esatto città
                limit=50,
                return_metadata=_meta,
            )
            return [_to_result(o) for o in resp.objects]

        # CASO B2: Filtro città + testo (es. "bar di Milano")
        if city_filter and query_text:
            combined = f"{query_text} {city_filter}"
            resp = collection.query.hybrid(
                query=combined,
                filters=agent_filter,
                alpha=0.5,
                limit=20,
                return_metadata=_meta,
            )
            return [_to_result(o) for o in resp.objects]

        # CASO C: Lista completa (nessun filtro testo)
        if not query_text or query_text.strip() == "*":
            resp = collection.query.fetch_objects(
                filters=agent_filter,
                limit=50,
            )
            return [_to_result(o) for o in resp.objects]

        # CASO D: Ricerca testo libero — hybrid search (BM25 + vector)
        resp = collection.query.hybrid(
            query=query_text,
            filters=agent_filter,
            alpha=0.5,  # bilanciato: keyword + semantico
            limit=20,
            return_metadata=_meta,
        )
        return [_to_result(o) for o in resp.objects]

    except Exception as e:
        print(f"❌ [WEAVIATE] Errore ricerca clienti: {e}")
        return []


# =============================================================================
# MAIN DI TEST
# =============================================================================
if __name__ == "__main__":
    # Carica .env per OPENAI_API_KEY
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".env",
        ))
    except ImportError:
        pass

    print("🚀 TEST RICERCA CLIENTI (WEAVIATE HYBRID)")
    tests = [
        ("AG001", "pizzeria Gigio", None),
        ("AG001", "Gigio", None),
        ("AG001", "mario rosi", None),      # typo
        
        ("AG001", "il caminetto", None),      # typo
        ("AG001", "al caminetto", None),      # typo
        ("AG001", "pub birra", None),        # semantico
        ("AG001", None, "Milano"),           # città
        ("AG001", "bar", "Milano"),          # testo + città
        ("AG001", "*", None),               # lista completa
    ]
    for agent, query, city in tests:
        res = search_client_smart(agent, query_text=query, city_filter=city)
        label = f"query={query!r} city={city!r}"
        if res:
            top = ", ".join(
                f"{r['alias']} ({r['client_id']}) [{r.get('score', '-')}]"
                for r in res[:5]
            )
            print(f"  ✅ {label} → {len(res)} risultati:\n     {top}")
            print("")
        else:
            print(f"  ❌ {label} → 0 risultati")
