import os
import json
import time
import logging
import hashlib
from typing import List, Dict, Tuple, Optional

from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, HttpResponseError
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    SearchFieldDataType,
    SearchField,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters
    
    
)
from azure.search.documents.indexes import SearchIndexClient

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings

# Basic retry helper
def retry_loop(fn, attempts=3, delay=2, backoff=2, exceptions=(Exception,), fn_name="operation"):
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except exceptions as e:
            last_exc = e
            logging.warning(f"{fn_name} failed (attempt {i+1}/{attempts}): {e}")
            time.sleep(delay * (backoff ** i))
    logging.error(f"{fn_name} failed after {attempts} attempts: {last_exc}")
    raise last_exc


class AzureSearchDataPipeline:
    def __init__(self, cfg: dict):

#Config-----------
        self.cfg = cfg

#Blob storage-------------
        if os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
            self.blob_service = BlobServiceClient.from_connection_string(cfg["azure"]["connection_string"])
        else:
            self.blob_service = BlobServiceClient(account_url=cfg["azure"]["blob_account_url"],
                                                  credential=DefaultAzureCredential())

        self.raw_container_name = cfg["azure"]["raw_container"]
        self.vector_container_name = cfg["azure"]["vector_container"]

#Vectorstore---------------
        self.index_name = cfg["search"]["index_name"]
        if cfg.get("search", {}).get("api_key"):
            self.search_client = SearchClient(endpoint=cfg["search"]["endpoint"],
                                              index_name=cfg["search"]["index_name"],
                                              credential=AzureKeyCredential(cfg["search"]["api_key"]))
        else:
            self.search_client = SearchClient(endpoint=cfg["search"]["endpoint"],
                                              index_name=cfg["search"]["index_name"],
                                              credential=DefaultAzureCredential())
#Embeddings-----------------
        openai_cfg = cfg["openai"]
        #---not for prod----
        # self.embeddings = AzureOpenAIEmbeddings(
        #     api_key=openai_cfg["api_key"],
        #     azure_endpoint=openai_cfg["endpoint"],
        #     azure_deployment=openai_cfg["deployment"],
        #     api_version=openai_cfg.get("api_version", "2023-05-15")
        # )
        self.embeddings = OpenAIEmbeddings(
            api_key=openai_cfg["api_key"],
            model=openai_cfg["embedding_model"],
        )
        self.vector_dimensions = len(self.embeddings.embed_query("hello world"))

#Chunking config-------------
        chunk_cfg = cfg.get("chunker", {})
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_cfg.get("chunk_size", 1000)),
            chunk_overlap=int(chunk_cfg.get("chunk_overlap", 200)),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        batch_cfg = cfg.get("batch", {})
        self.embed_batch_size = int(batch_cfg.get("embed_batch_size", 32))
        self.upsert_batch_size = int(batch_cfg.get("upsert_batch_size", 100))

        self.hash_blob_name = "hashes.json"
        logging.info("AzureSearchRagPipeline initialized")

    def compute_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def load_hashes(self) -> set:
        """Load set of processed file-hashes from vector container's hashes.json. Return empty set if not present."""
        container = self.blob_service.get_container_client(self.vector_container_name)
        try:
            blob = container.get_blob_client(self.hash_blob_name)
            raw = blob.download_blob().readall()
            return set(json.loads(raw.decode("utf-8")))
        except ResourceNotFoundError:
            return set()
        except Exception as e:
            logging.warning(f"Could not load hashes.json: {e}")
            return set()

    def save_hashes(self, hashes: set):
        container = self.blob_service.get_container_client(self.vector_container_name)
        try:
            container.create_container()
        except ResourceExistsError:
            pass
        blob = container.get_blob_client(self.hash_blob_name)
        blob.upload_blob(json.dumps(list(hashes)), overwrite=True)

    def scan_for_new_chunks(self) -> Tuple[List[str], List[Dict], set]:
        """
        Scan raw container, return (chunks, metadatas, new_hashes)
        - chunks: list of chunk text strings
        - metadatas: list of dicts aligned with chunks (metadata, chunk_index, hash)
        - new_hashes: set of file-level hashes to be appended to stored hashes after successful upsert
        """
        raw = self.blob_service.get_container_client(self.raw_container_name)
        known_hashes = self.load_hashes()
        chunks: List[str] = []
        metadatas: List[Dict] = []
        new_hashes = set()

        for blob_props in raw.list_blobs():
            logging.info(f"Checking blob {blob_props.name}")
            bclient = raw.get_blob_client(blob_props.name)
            try:
                data = bclient.download_blob().readall()
            except Exception as e:
                logging.warning(f"Failed to download blob {blob_props.name}: {e}")
                continue

            file_hash = self.compute_hash(data)
            if file_hash in known_hashes:
                logging.debug(f"Skipping already-processed blob {blob_props.name}")
                continue

            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                logging.warning(f"Blob {blob_props.name} is not decodable as utf-8; skipping")
                continue

            file_chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(file_chunks):
                chunks.append(chunk)
                metadatas.append({
                    "metadata": blob_props.name,
                    "chunk_index": i,
                    "hash": file_hash
                })
            new_hashes.add(file_hash)
            logging.info(f"Collected {len(file_chunks)} chunks from {blob_props.name}")

        return chunks, metadatas, new_hashes

    def embed_batches(self, texts: List[str]) -> List[List[float]]:
        """Embed in batches using embeddings.embed_documents (returns list of vectors)."""
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            logging.info(f"Embedding batch {i//self.embed_batch_size + 1}: {len(batch)} items")
            
            def call_embed():
                return self.embeddings.embed_documents(batch)
            vecs = retry_loop(call_embed, attempts=4, delay=1, backoff=2, exceptions=(Exception,), fn_name="embed_documents")
            vectors.extend(vecs)
            time.sleep(0.05) 
        return vectors

    def build_documents_for_search(self, chunks: List[str], metas: List[Dict], vectors: List[List[float]]) -> List[Dict]:
        docs = []
        for chunk, meta, vec in zip(chunks, metas, vectors):
            doc_id = f"{meta['hash']}_{meta['chunk_index']}"
            docs.append({
                "id": doc_id,
                "content": chunk,
                "metadata": meta["metadata"],
                "chunk_index": meta["chunk_index"],
                "hash": meta["hash"],
                "content_vector": vec
            })
        return docs



    def upsert_batches(self, docs: List[Dict]):
        """Upload docs to Azure Search in batches, creating the index if needed and retrying on failures."""
        for i in range(0, len(docs), self.upsert_batch_size):
            batch = docs[i:i + self.upsert_batch_size]
            logging.info(f"Upserting batch {i // self.upsert_batch_size + 1}: {len(batch)} docs")

            # def call_upload():
            #     return self.search_client.upload_documents(documents=batch)
            try:
                # results = retry_loop(
                #     call_upload,
                #     attempts=3,
                #     delay=1,
                #     backoff=2,
                #     exceptions=(HttpResponseError,),
                #     fn_name="upload_documents"
                # )
                results = self.search_client.upload_documents(documents=batch)

            except ResourceNotFoundError as e:
                logging.warning(f"⚠️ Index not found. Attempting to create index: {self.index_name}")
                self._create_index_if_missing()
                logging.info("Retrying upload after index creation...")
                time.sleep(2)  # small delay before retry
                try:
                    results = self.search_client.upload_documents(documents=batch)
                except Exception as e:
                    name = self.search_client._index_name
                    logging.error(f"❌ Failed to upload documents after index creation: {e}",name)
                    raise e




            # Process results (IndexingResult)
            for res in results:
                if not getattr(res, "succeeded", True):
                    logging.error(
                        f"❌ Indexing failed for key {getattr(res, 'key', None)}: "
                        f"{getattr(res, 'error_message', None)}"
                    )




    def _create_index_if_missing(self):
        index_client = SearchIndexClient(endpoint=self.cfg["search"]["endpoint"], credential=AzureKeyCredential(self.cfg["search"]["api_key"]))

        try:
            index_client.get_index(self.index_name)
            logging.info(f"✅ Index '{self.index_name}' already exists:{index_client}")
            return
        except ResourceNotFoundError:
            logging.warning(f"⚠️ Index '{self.index_name}' not found. Creating it now...")
            vector_search = VectorSearch(
                algorithms=[
                    # VectorSearchAlgorithmConfiguration(
                    #     name="default-hnsw",    # 1️⃣ Algorithm name
                    #     kind="hnsw"
                    # ),
                    HnswAlgorithmConfiguration(
                        name="default-hnsw",
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric="cosine"
                        )
                    )
                ],

                profiles=[
                    VectorSearchProfile(
                        name="default-profile",  
                        algorithm_configuration_name="default-hnsw",
                    )
                ]
            )
            # Define the index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True,filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String, searchable=True,),
                SimpleField(name="metadata", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="hash", type=SearchFieldDataType.String, filterable=True),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.vector_dimensions,  # set dynamically from embeddings
                    vector_search_profile_name="default-profile"
                    
                ),

            ]


            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )

            # Create the index
            index_client.create_index(index)
            logging.info(f"✅ Index '{self.index_name}' created successfully.")

            # Wait for index to become available
            for i in range(10):
                try:
                    time.sleep(10)
                    index_client.get_index(self.index_name)
                    logging.info(f"✅ Index '{self.index_name}' is now active.")
                    return
                except ResourceNotFoundError:
                    logging.info("⏳ Waiting for index to become available...")
                    time.sleep(2)

            raise RuntimeError("❌ Index creation timed out — still not available.")



    def run_update(self):
        """Scan, embed, and upsert new chunks to Azure Cognitive Search, then persist hashes."""
        logging.info("Starting pipeline run_update...")
        chunks, metas, new_hashes = self.scan_for_new_chunks()
        if not chunks:
            logging.info("No new chunks found. Exiting.")
            return

        # Get embeddings
        vectors = self.embed_batches(chunks)
        if len(vectors) != len(chunks):
            logging.error("Number of vectors does not match chunks - aborting")
            raise RuntimeError("Embedding length mismatch")

        docs = self.build_documents_for_search(chunks, metas, vectors)
        # Upsert
        self.upsert_batches(docs)

        # Persist hashes after a successful upsert
        all_hashes = self.load_hashes()
        all_hashes.update(new_hashes)
        self.save_hashes(all_hashes)
        logging.info(f"Run complete. Indexed {len(docs)} chunks from {len(new_hashes)} new files.")













