"""
Gestione collection Weaviate "Clienti".

Comandi:
    python scripts/weaviate_setup.py create   # Crea collection
    python scripts/weaviate_setup.py load     # Carica dati da data/clienti.csv
    python scripts/weaviate_setup.py delete   # Cancella collection
    python scripts/weaviate_setup.py reset    # Delete + Create + Load
    python scripts/weaviate_setup.py count    # Conta oggetti nella collection
"""

import csv
import os
import sys
import time

import weaviate.classes.config as wc

# Percorsi
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "clienti.csv")

# Aggiungi root al path per importare weaviate_client
sys.path.insert(0, PROJECT_ROOT)
from src.tools.weaviate_client import get_weaviate_client, COLLECTION_NAME


def create_collection():
    """Crea la collection Clienti con schema e vectorizer text2vec-openai."""
    client = get_weaviate_client()

    if client.collections.exists(COLLECTION_NAME):
        print(f"⚠️  Collection '{COLLECTION_NAME}' esiste già — skip create")
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
            model="text-embedding-3-small",
        ),  # TODO: migrare a vector_config quando si aggiorna weaviate-client
        properties=[
            wc.Property(name="client_id",      data_type=wc.DataType.TEXT,
                        skip_vectorization=True, tokenization=wc.Tokenization.FIELD),
            wc.Property(name="ragione_sociale", data_type=wc.DataType.TEXT),
            wc.Property(name="alias",           data_type=wc.DataType.TEXT),
            wc.Property(name="agent_id",        data_type=wc.DataType.TEXT,
                        skip_vectorization=True, tokenization=wc.Tokenization.FIELD),
            wc.Property(name="indirizzo",       data_type=wc.DataType.TEXT),
            wc.Property(name="citta",           data_type=wc.DataType.TEXT),
        ],
    )
    print(f"✅ Collection '{COLLECTION_NAME}' creata")


def load_data():
    """Carica clienti da data/clienti.csv nella collection Weaviate."""
    client = get_weaviate_client()

    if not client.collections.exists(COLLECTION_NAME):
        print(f"❌ Collection '{COLLECTION_NAME}' non esiste — esegui prima 'create'")
        return

    if not os.path.exists(CSV_PATH):
        print(f"❌ File non trovato: {CSV_PATH}")
        return

    collection = client.collections.get(COLLECTION_NAME)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("⚠️  CSV vuoto")
        return

    # Insert singolo (REST) con retry — funziona anche senza gRPC (es. Azure esterno)
    MAX_RETRIES = 3
    for i, row in enumerate(rows, 1):
        props = {
            "client_id":        row["client_id"],
            "ragione_sociale":  row["ragione_sociale"],
            "alias":            row["alias"],
            "agent_id":         row["agent_id"],
            "indirizzo":        row["indirizzo"],
            "citta":            row["citta"],
        }
        for attempt in range(MAX_RETRIES):
            try:
                collection.data.insert(properties=props)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  ⚠️  Retry {attempt+1}/{MAX_RETRIES} per {row['client_id']} "
                          f"(attendo {wait}s): {e}")
                    time.sleep(wait)
                else:
                    print(f"  ❌ Fallito insert {row['client_id']} dopo {MAX_RETRIES} tentativi: {e}")
                    raise
        print(f"  [{i}/{len(rows)}] {row['client_id']} — {row['alias']}")

    count = collection.aggregate.over_all(total_count=True).total_count
    print(f"✅ {count} clienti caricati nella collection '{COLLECTION_NAME}'")


def delete_collection():
    """Cancella la collection Clienti."""
    client = get_weaviate_client()

    if not client.collections.exists(COLLECTION_NAME):
        print(f"⚠️  Collection '{COLLECTION_NAME}' non esiste — nulla da cancellare")
        return

    client.collections.delete(COLLECTION_NAME)
    print(f"🗑️  Collection '{COLLECTION_NAME}' cancellata")


def count_objects():
    """Conta gli oggetti nella collection."""
    client = get_weaviate_client()

    if not client.collections.exists(COLLECTION_NAME):
        print(f"❌ Collection '{COLLECTION_NAME}' non esiste")
        return

    collection = client.collections.get(COLLECTION_NAME)
    n = collection.aggregate.over_all(total_count=True).total_count
    print(f"📊 Collection '{COLLECTION_NAME}': {n} oggetti")


def reset_collection():
    """Delete + Create + Load."""
    delete_collection()
    create_collection()
    load_data()


# --- CLI ---
COMMANDS = {
    "create": create_collection,
    "load":   load_data,
    "delete": delete_collection,
    "reset":  reset_collection,
    "count":  count_objects,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Uso: python {sys.argv[0]} <{'|'.join(COMMANDS)}>")
        sys.exit(1)

    # Carica .env per OPENAI_API_KEY e WEAVIATE_URL
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    except ImportError:
        pass

    COMMANDS[sys.argv[1]]()
