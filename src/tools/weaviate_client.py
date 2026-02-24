"""Weaviate client singleton — connessione lazy e collection helper."""

import os
import weaviate

_client = None
COLLECTION_NAME = "Clienti"


def get_weaviate_client():
    """Restituisce un client Weaviate connesso (lazy init, riconnette se persa)."""
    global _client
    if _client is None or not _client.is_connected():
        url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        secure = url.startswith("https")
        host = url.replace("https://", "").replace("http://", "").split(":")[0]

        # Porta HTTP: esplicita dall'URL, altrimenti 443 (HTTPS) o 8080 (HTTP locale)
        url_after_scheme = url.rsplit("//", 1)[-1]
        if ":" in url_after_scheme:
            port = int(url_after_scheme.split(":")[-1].rstrip("/"))
        else:
            port = 443 if secure else 8080

        grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        grpc_host = os.getenv("WEAVIATE_GRPC_HOST", host)

        # skip_init_checks: WEAVIATE_SKIP_INIT=true per Azure (gRPC health check non passa)
        skip = os.getenv("WEAVIATE_SKIP_INIT", "true" if secure else "false").lower() == "true"

        _client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=secure,
            headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")},
            skip_init_checks=skip,
        )
    return _client


def get_collection():
    """Restituisce la collection Clienti."""
    return get_weaviate_client().collections.get(COLLECTION_NAME)
