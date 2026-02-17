import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)

# 1. Caricamento variabili (prova i percorsi comuni se non trova il .env)
load_dotenv() # Cerca nella cartella corrente
if not os.getenv("AZURE_SEARCH_ADMIN_KEY"):
    load_dotenv("../.env") # Cerca una cartella sopra

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = "anagrafica_bevande"

# Controllo sicurezza
if not endpoint or not key:
    print(f"ERRORE: Variabili non trovate!")
    print(f"Endpoint: {endpoint}")
    print(f"Key: {'Trovata' if key else 'NON TROVATA'}")
    exit(1)

client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def create_beverage_index():
    print(f"Eliminazione indice (se esiste): {index_name}...")
    try:
        client.delete_index(index_name)
    except Exception:
        pass

    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True, retrievable=True),
        SearchField(name="sku", type=SearchFieldDataType.String, searchable=True, filterable=True, retrievable=True),
        SearchField(name="descrizione", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchField(name="brand", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True, retrievable=True),
        SearchField(name="categoria", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True, retrievable=True),
        SearchField(name="sottocategoria", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True, retrievable=True),
        SearchField(name="famiglia", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True, retrievable=True),
        SearchField(name="formato", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True, retrievable=True),
        SearchField(name="confezione", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True, retrievable=True),
        SearchField(name="prezzo_unitario", type=SearchFieldDataType.Double, filterable=True, sortable=True, retrievable=True),
        SearchField(name="prezzo_confezione", type=SearchFieldDataType.String, searchable=True, filterable=True, sortable=True, retrievable=True),
        SearchField(name="disponibilita", type=SearchFieldDataType.Double, filterable=True, retrievable=True),
        SearchField(name="note_ricerca", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True, retrievable=True),
        
        SearchField(
            name="descrizione_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="anagrafica_bevande_ss",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="descrizione"),
                    content_fields=[SemanticField(field_name="descrizione")],
                    keywords_fields=[
                        SemanticField(field_name="brand"),
                        SemanticField(field_name="categoria"),
                        SemanticField(field_name="note_ricerca")
                    ]
                )
            )
        ]
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    print(f"Creazione nuovo indice {index_name}...")
    client.create_index(index)
    print("✅ Indice creato con successo!")

if __name__ == "__main__":
    create_beverage_index()