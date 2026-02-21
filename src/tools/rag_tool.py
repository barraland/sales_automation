## src.tools.rag_tool.py
import os
import json
import logging
from typing import List, Dict, Any, Optional

from langchain.tools import tool
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag_tool")
logger.setLevel(logging.INFO)

AZ_SEARCH_INDEX = "anagrafica_bevande" 
AZ_SEMANTIC_CONFIG = "anagrafica_bevande_ss" 
AZ_VECTOR_FIELD = "descrizione_vector"
EMBED_MODEL = "text-embedding-3-small"

_search_client_cache = None
_index_client_cache = None
_openai_client_cache = None
_filterable_fields_cache = None

def get_search_client():
    global _search_client_cache
    if _search_client_cache is None:
        _search_client_cache = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", "").strip(),
            index_name=AZ_SEARCH_INDEX,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY", "").strip())
        )
    return _search_client_cache

def get_index_client():
    global _index_client_cache
    if _index_client_cache is None:
        _index_client_cache = SearchIndexClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", "").strip(),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY", "").strip())
        )
    return _index_client_cache

def get_openai_client():
    global _openai_client_cache
    if _openai_client_cache is None:
        _openai_client_cache = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())
    return _openai_client_cache

def get_filterable_fields():
    global _filterable_fields_cache
    if _filterable_fields_cache is None:
        try:
            idx = get_index_client().get_index(AZ_SEARCH_INDEX)
            _filterable_fields_cache = {str(f.name) for f in idx.fields if f.filterable}
        except Exception:
            _filterable_fields_cache = {"sku", "brand", "categoria", "prezzo_unitario", "disponibilita", "formato"}
    return _filterable_fields_cache

# ---------------------------------------------------------
# Helper Functions - CORRETTA PER ESCAPING APOSTROFI
# ---------------------------------------------------------

def _build_odata_filter(filters_json: str) -> Optional[str]:
    if not filters_json:
        return None
    try:
        filters = json.loads(filters_json)
        odata_parts = []
        allowed = get_filterable_fields()
        for field, value in filters.items():
            if field not in allowed: continue
            
            if isinstance(value, dict):
                for op, val in value.items():
                    if op in ["gt", "lt", "ge", "le"]:
                        odata_parts.append(f"{field} {op} {val}")
            elif isinstance(value, str):
                # ESCAPING: Sostituiamo ' con '' per OData
                escaped_val = value.replace("'", "''")
                odata_parts.append(f"{field} eq '{escaped_val}'")
            else:
                odata_parts.append(f"{field} eq {value}")
        return " and ".join(odata_parts) if odata_parts else None
    except:
        return None

def _embed_query(text: str) -> List[float]:
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def _azure_search_retrieve(query: str, top_k: int, odata_filter: Optional[str]) -> List[Dict[str, Any]]:
    qvec = _embed_query(query)
    vector_query = VectorizedQuery(vector=qvec, k_nearest_neighbors=top_k, fields=AZ_VECTOR_FIELD)
    
    results = get_search_client().search(
        search_text=query,
        top=top_k,
        filter=odata_filter,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name=AZ_SEMANTIC_CONFIG
    )
    return [dict(r) for r in results]

# ---------------------------------------------------------
# LLM Reranker
# ---------------------------------------------------------

def _llm_rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filtra i candidati RAG con un LLM: per ogni prodotto decide se è
    plausibilmente ciò che l'utente cerca (sì/no strutturato).
    Ritorna i candidati "sì". Se tutti vengono scartati, ritorna tutti (fallback).
    """
    if len(candidates) <= 1:
        return candidates

    client = get_openai_client()

    cand_lines = "\n".join(
        f"{i+1}. SKU={c.get('sku')} | Brand={c.get('brand')} | "
        f"Descrizione={c.get('descrizione')} | Formato={c.get('formato')} | "
        f"Categoria={c.get('categoria')}"
        for i, c in enumerate(candidates)
    )

    prompt = (
        f"Sei un assistente catalogo bevande Horeca. Un agente commerciale ha cercato: \"{query}\"\n\n"
        f"Il motore di ricerca Azure AI Search ha restituito questi candidati:\n{cand_lines}\n\n"
        f"Per ogni candidato, decidi se è plausibile che sia il prodotto cercato dall'agente.\n\n"
        f"REGOLE:\n"
        f"- Sii tollerante a typo e varianti ortografiche "
        f"(es. 'Ichnuza non filtra' = 'Ichnusa non filtrata', 'Martni' = 'Martini')\n"
        f"- PRIORITÀ MASSIMA a brand e tipologia: se l'utente cerca 'Martini Rosso', "
        f"solo prodotti del brand Martini & Rossi sono validi — altri brand che contengono "
        f"'rosso' nel nome NON sono accettabili\n"
        f"- Formato/volume diverso dello stesso prodotto È accettabile "
        f"(es. 33cl vs 50cl della stessa birra), a meno che il formato non sia parte della query utente.\n"
        f"- Se la query indica solo il brand senza tipologia specifica, includi tutte le varianti\n\n"
        f"Rispondi SOLO con un JSON array: "
        f'[{{"sku": "SKU1", "match": true}}, {{"sku": "SKU2", "match": false}}, ...]'
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = json.loads(resp.choices[0].message.content)
        # json_object mode wrappa in dict — estrai la lista
        items = raw if isinstance(raw, list) else next(
            (v for v in raw.values() if isinstance(v, list)), []
        )
        match_skus = {item["sku"] for item in items if item.get("match")}
        filtered = [c for c in candidates if c.get("sku") in match_skus]
        kept = filtered if filtered else candidates   # fallback: non eliminare tutto
        print(f"   🤖 [RERANK] {len(candidates)} → {len(kept)} candidati")
        return kept
    except Exception as e:
        print(f"   ⚠️ [RERANK ERROR]: {e}")
        return candidates


# ---------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------
@tool
def search_product_smart(
    query: str,
    planner_motivation: str,
    filters_json: str = "",
    top_k: int = 10,
    sku_whitelist: Optional[List[str]] = None,
    llm_rerank: bool = True,
) -> Dict[str, Any]:
    """
    Cerca prodotti nel catalogo bevande.
    Restituisce dati tecnici e metadati sulla ricerca effettuata.
    """
    odata_filter = _build_odata_filter(filters_json)

    # Se è presente una whitelist di SKU (storico ordini cliente), applica il filtro
    if sku_whitelist:
        sku_values = "|".join(sku_whitelist)
        sku_filter = f"search.in(sku, '{sku_values}', '|')"
        if odata_filter:
            odata_filter = f"({odata_filter}) and ({sku_filter})"
        else:
            odata_filter = sku_filter

    if not query or not query.strip():
        query = "bevande" # O un termine generico che non rompa l'embedding

    print(f"\n🔍 [RAG CALL]")
    print(f"   ├─ Motivation: {planner_motivation}")
    print(f"   ├─ Query: {query}")
    print(f"   ├─ Filters: {filters_json if filters_json else 'None'}")
    print(f"   ├─ SKU whitelist: {len(sku_whitelist) if sku_whitelist else 0} SKU")
    print(f"   ├─ LLM rerank: {llm_rerank and not sku_whitelist}")
    print(f"   └─ K: {top_k}")

    try:
        docs = _azure_search_retrieve(query, top_k, odata_filter)

        results = []
        found_brands = set()

        for d in docs:
            found_brands.add(str(d.get("brand", "Unknown")))
            results.append({
                "sku": d.get("sku"),
                "descrizione": d.get("descrizione"),
                "brand": d.get("brand"),
                "formato": d.get("formato"),
                "prezzo": d.get("prezzo_unitario"),
                "stock": d.get("disponibilita"),
                "categoria": d.get("categoria")
            })

        # LLM reranking: filtra candidati non pertinenti (skip se sku_whitelist già filtra)
        if llm_rerank and not sku_whitelist and results:
            results = _llm_rerank(query, results)
            found_brands = {c.get("brand", "Unknown") for c in results}

        rag_motivation = ""
        if len(results) > 0:
            rag_motivation = f"Ho trovato {len(results)} prodotti corrispondenti dei brand: {', '.join(found_brands)}."
        else:
            rag_motivation = "Nessun prodotto trovato nel database per questa specifica combinazione."

        print(f"✅ [RAG RESULT]")
        print(f"   └─ {rag_motivation}")

        return {
            "results": results,
            "metadata": {
                "planner_intent": planner_motivation,
                "rag_summary": rag_motivation,
                "query_used": query,
                "filters_used": odata_filter
            }
        }

    except Exception as e:
        print(f"❌ [RAG ERROR]: {e}")
        return {"results": [], "metadata": {"error": str(e)}}



# ---------------------------------------------------------
# Facet Metadata Reader (pubblica - usata dal planner)
# ---------------------------------------------------------

def get_catalog_facets(
    facet_fields: Optional[List[str]] = None,
    max_values_per_facet: int = 100,
    odata_filter: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Recupera valori distinti di brand, categoria, ecc. direttamente dalle FACETS
    dell'indice Azure Search (senza scansione documenti).

    Args:
        facet_fields: lista campi su cui calcolare le facets.
                      Default: ["brand", "categoria"]
        max_values_per_facet: numero massimo valori per facet
        odata_filter: filtro OData opzionale (es. "categoria eq 'Birra'" per
                      recuperare solo i brand di una certa categoria)

    Returns:
        {
            "brand": [...],
            "categoria": [...]
        }
    """

    if facet_fields is None:
        facet_fields = ["brand", "categoria"]

    search_client = get_search_client()

    # Costruiamo sintassi facets Azure: "brand,count:100"
    facets_query = [
        f"{field},count:{max_values_per_facet}"
        for field in facet_fields
    ]

    try:
        results = search_client.search(
            search_text="*",        # nessun filtro semantico
            top=0,                  # non vogliamo documenti
            facets=facets_query,
            filter=odata_filter
        )

        facet_data = results.get_facets()

        output = {}

        for field in facet_fields:
            values = []
            if field in facet_data:
                for item in facet_data[field]:
                    if item.get("value") is not None:
                        values.append(str(item["value"]))

            output[field] = sorted(values)

        return output

    except Exception as e:
        logger.error(f"Errore nel recupero facets: {e}")
        return {field: [] for field in facet_fields}


# ---------------------------------------------------------
# Main per il Test
# ---------------------------------------------------------
if __name__ == "__main__":
    
    
    
    
    print("\n" + "="*50)
    print("🚀 TEST RAG TOOL - DEBUG MODE")
    print("="*50)
    
    # Test 1: Ricerca puntuale per Brand (con test escaping apostrofo)
    test_query = "Becks"
    test_motivation = "L'utente ha chiesto esplicitamente quali Beck's sono disponibili dopo aver visto il brand nell'anagrafica."
    test_filters = json.dumps({"brand": "Beck's"})

    print(f"\n👉 Eseguo Test 1 (Filtro Brand con apostrofo)...")
    output = search_product_smart.invoke({
        "query": test_query,
        "planner_motivation": test_motivation,
        "filters_json": test_filters,
        "top_k": 5
    })

    print("\n--- OUTPUT RAW (DATI GREZZI PYTHON) ---")
    print(output)

    print("\n--- OUTPUT STRUTTURATO (JSON FORMATTED) ---")
    print(json.dumps(output, indent=2))

    print("\n" + "-"*50)

    
    
    # Test 2: Ricerca senza filtri
    print(f"\n👉 Eseguo Test 2 (Senza filtri - Analcolica)...")
    output_simple = search_product_smart.invoke({
        "query": "birra analcolica",
        "planner_motivation": "L'utente cerca opzioni senza alcol, controllo disponibilità generica.",
        "top_k": 3
    })
    
    print("\n--- OUTPUT RAW (DATI GREZZI PYTHON) ---")
    print(output_simple)

    print(f"\n✅ Fine Test. Prodotti trovati: {len(output_simple['results'])}")
    print(f"Sintesi RAG: {output_simple['metadata']['rag_summary']}")
    print("\n" + "-"*50)

    
    
    # Test 3: Recupero Facets (Brand e Categoria)
    print(f"\n👉 Eseguo Test 3 (Recupero Facets Catalogo)...")

    facets_output = get_catalog_facets()

    print("\n--- FACETS OUTPUT (VALORI DISTINTI DA AZURE) ---")
    print(json.dumps(facets_output, indent=2))

    print(f"\n✅ Brand trovati: {len(facets_output.get('brand', []))}")
    print(f"✅ Categorie trovate: {len(facets_output.get('categoria', []))}")
