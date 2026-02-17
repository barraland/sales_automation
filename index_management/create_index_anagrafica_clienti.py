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

# 1. Caricamento variabili
load_dotenv()
if not os.getenv("AZURE_SEARCH_ADMIN_KEY"):
    load_dotenv("../.env") # Cerca una cartella sopra
    
    
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = "anagrafica_clienti"

if not endpoint or not key:
    print(f"ERRORE: Variabili non trovate!")
    exit(1)

client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def create_client_index():
    print(f"Eliminazione indice (se esiste): {index_name}...")
    try:
        client.delete_index(index_name)
    except Exception:
        pass

    fields = [
        # ID univoco del record (necessario per Azure Search)
        SearchField(name="id", type=SearchFieldDataType.String, key=True, retrievable=True),
        
        # ID del cliente nel tuo ERP
        SearchField(name="client_id", type=SearchFieldDataType.String, searchable=True, filterable=True, retrievable=True),
        
        # Ragione Sociale (es. "Mario Rossi S.r.l.")
        SearchField(name="ragione_sociale", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        
        # Alias per gestire i nomi colloquiali (es. "Bar Baffo")
        SearchField(name="alias", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        
        # Partita IVA o Codice Fiscale
        SearchField(name="piva_cf", type=SearchFieldDataType.String, searchable=True, filterable=True, retrievable=True),
        
        # ID dell'Agente (fondamentale per filtrare i clienti visibili all'utente corrente)
        SearchField(name="agent_id", type=SearchFieldDataType.String, filterable=True, facetable=True, retrievable=True),
        
        # Indirizzi di destinazione: memorizzati come stringa JSON o testo semplice
        # In Azure Search, una Collection di stringhe è ottima per gestire indirizzi multipli
        SearchField(name="destinazioni", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True, retrievable=True),
        
        # Campi per ricerca vettoriale (per trovare clienti simili semanticamente)
        SearchField(
            name="ragione_sociale_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]

    # Configurazione ricerca vettoriale
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )

    # Configurazione ricerca semantica (per capire "Bar Mario" anche con typo)
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="anagrafica_clienti_ss",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="ragione_sociale"),
                    content_fields=[
                        SemanticField(field_name="alias"),
                        SemanticField(field_name="destinazioni")
                    ],
                    keywords_fields=[
                        SemanticField(field_name="client_id"),
                        SemanticField(field_name="piva_cf")
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
    print("✅ Indice ANAGRAFICA_CLIENTI creato con successo!")

if __name__ == "__main__":
    create_client_index()