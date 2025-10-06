import logging
import os
from DataPipeline.AzureDataPipeline import AzureSearchRagPipeline
import azure.functions as func 
logging.basicConfig(level=logging.INFO)

cfg = {
  "azure": {
    "blob_account_url": os.getenv("AZURE_BLOB_ACCOUNT_URL"),
    "raw_container": os.getenv("AZURE_RAW_CONTAINER", "raw-data"),
    "vector_container": os.getenv("AZURE_VECTOR_CONTAINER", "vector-index-container")
  },
  "search": {
    "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
    "index_name": os.getenv("AZURE_SEARCH_INDEX"),
    "api_key": os.getenv("AZURE_SEARCH_API_KEY", "")
  },
  "openai": {
    "endpoint": os.getenv("OPENAI_AZURE_ENDPOINT"),
    "api_key": os.getenv("OPENAI_AZURE_KEY"),
    "deployment": os.getenv("OPENAI_DEPLOYMENT"),
    "api_version": os.getenv("OPENAI_API_VERSION", "2023-05-15")
  },
  "chunker": { "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")), "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")) },
  "batch": { "embed_batch_size": int(os.getenv("EMBED_BATCH_SIZE", "32")), "upsert_batch_size": int(os.getenv("UPSERT_BATCH_SIZE", "100")) }
}
def main(mytimer: func.TimerRequest) -> None:
    logging.info("Timer triggered — starting pipeline...")

    pipeline = AzureSearchRagPipeline(cfg)
    pipeline.run_update()
    
    logging.info("Pipeline finished successfully")

