import os
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# -------------------------------------------------
# Load env
# -------------------------------------------------
load_dotenv()

AZ_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip()
AZ_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY", "").strip()
AZ_SEARCH_INDEX = 'anagrafica_bevande'

if not AZ_SEARCH_ENDPOINT or not AZ_SEARCH_ADMIN_KEY:
    raise RuntimeError("Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_ADMIN_KEY")

# -------------------------------------------------
# Search client
# -------------------------------------------------
search = SearchClient(
    endpoint=AZ_SEARCH_ENDPOINT,
    index_name=AZ_SEARCH_INDEX,
    credential=AzureKeyCredential(AZ_SEARCH_ADMIN_KEY),
)

# -------------------------------------------------
# Collect document IDs
# -------------------------------------------------
ids = []

results = search.search(
    search_text="*",
    select=["id"],
    top=1000,  # max per page
)

for r in results:
    ids.append({"id": r["id"]})

print(f"Found {len(ids)} documents")
print(ids[:5])  # preview

if ids:
    search.delete_documents(documents=ids)
    print(f"🧹 Deleted {len(ids)} documents")