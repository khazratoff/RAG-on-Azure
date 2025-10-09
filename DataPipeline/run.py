import logging
import os
from AzureDataPipeline import AzureSearchDataPipeline
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO)

load_dotenv()

data_cfg = {
  "azure": {
    "blob_account_url": os.getenv("AZURE_BLOB_ACCOUNT_URL"),
    "connection_string": os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    "raw_container": os.getenv("AZURE_RAW_CONTAINER", "raw-docs"),
    "vector_container": os.getenv("AZURE_VECTOR_CONTAINER", "vector-indexes")
  },
  "search": {
    "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
    "index_name": os.getenv("AZURE_SEARCH_INDEX"),
    "api_key": os.getenv("AZURE_SEARCH_API_KEY", "")
  },
  "openai": {
    # "endpoint": os.getenv("OPENAI_AZURE_ENDPOINT"),
    "api_key": os.getenv("OPENAI_API_KEY"),
    "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
    # "deployment": os.getenv("OPENAI_DEPLOYMENT"),
    # "api_version": os.getenv("OPENAI_API_VERSION", "2023-05-15")
  },
  "chunker": { "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")), "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")) },
  "batch": { "embed_batch_size": int(os.getenv("EMBED_BATCH_SIZE", "32")), "upsert_batch_size": int(os.getenv("UPSERT_BATCH_SIZE", "100")) }
}


def main() -> None:
    logging.info("Timer triggered — starting pipeline...")

    pipeline = AzureSearchDataPipeline(data_cfg)
    pipeline.run_update()
    
    logging.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()