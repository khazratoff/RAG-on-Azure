import logging
import hashlib
import os
import json
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from langchain_openai import  AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS



class RagDataPipeline():

    def __init__(self, config):

        self.config = config
        self.account_url = config.azure.account_url
        self.default_credential = DefaultAzureCredential()
        self.embeddings = AzureOpenAIEmbeddings(model=config.embedding.model,api_key=config.azure.openai_key,azure_endpoint="https://ai-proxy.lab.epam.com")
        self.HASH_FILE = "hashes.json"
        self.blob_service = BlobServiceClient(account_url=self.account_url, credential=self.default_credential)


    def compute_hash(self,content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
    

    def upload_directory(self,local_dir: str, container_name: str):
        """Upload all files in a directory to a blob container."""
        container_client = self.blob_service.get_container_client(container_name)
        container_client.create_container(exist_ok=True)
        for fname in os.listdir(local_dir):
            fpath = os.path.join(local_dir, fname)
            blob_client = container_client.get_blob_client(fname)
            with open(fpath, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)

    def download_directory(self,container_name: str, local_dir: str):
        """Download all files from a blob container into a local directory."""
        os.makedirs(local_dir, exist_ok=True)
        container_client = self.blob_service.get_container_client(container_name)
        for blob in container_client.list_blobs():
            blob_client = container_client.get_blob_client(blob.name)
            local_path = os.path.join(local_dir, blob.name)
            with open(local_path, "wb") as f:
                f.write(blob_client.download_blob().readall())


    def extract_text(self):

        pass


    def load_hashes(self,index_container) -> set:
        """Download hashes.json from blob (if exists)."""
        try:
            blob_client = index_container.get_blob_client(self.HASH_FILE)
            data = blob_client.download_blob().readall()
            return set(json.loads(data.decode("utf-8")))
        except Exception:
            return set()

    def save_hashes(self, hashes: set):
        """Upload hashes.json to blob."""
        container_client = self.blob_service.get_container_client(self.config.azure.storage.container.vector)
        blob_client = container_client.get_blob_client(self.HASH_FILE)
        blob_client.upload_blob(json.dumps(list(hashes)), overwrite=True)



    def scan(self):
        """Scan raw-data container and return only new (deduped) docs."""
        raw_container = self.blob_service.get_container_client(self.config.azure.storage.container.raw)
        index_container = self.blob_service.get_container_client(self.config.azure.storage.container.vector)
        known_hashes = self.load_hashes(index_container)
        new_docs, new_hashes = [], set()

        for blob in raw_container.list_blobs():
            blob_client = raw_container.get_blob_client(blob)
            content = blob_client.download_blob().readall()
            file_hash = self.compute_hash(content)

            if file_hash not in known_hashes:
                # Store plain text here (adjust if PDF, etc.)
                new_docs.append(content.decode("utf-8"))
                new_hashes.add(file_hash)

        return new_docs, new_hashes


    def update_vectorstore(self):

        index_container = self.blob_service.get_container_client(self.config.azure.storage.container.vector)
        new_docs, new_hashes = self.scan()

        if not new_docs:
            print("ℹ️ No new docs to embed — skipping FAISS update")
            return



        # Load or create FAISS
        local_vectorstore = self.config.vectorstore.path
        if any(b.name == "index.faiss" for b in index_container.list_blobs()):
            self.download_directory(self.config.azure.storage.container.vector, local_vectorstore)
            vectorstore = FAISS.load_local(local_vectorstore, self.embeddings, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS(embedding_function=self.embeddings, index=None, docstore=None, index_to_docstore_id={})

        # Add new docs
        vectorstore.add_texts(new_docs)

        # Save updated FAISS
        vectorstore.save_local(local_vectorstore)
        self.upload_directory(local_vectorstore,self.config.azure.storage.container.vector)

        # Save updated hashes
        hashes = self.load_hashes(index_container)
        hashes.update(new_hashes)
        self.save_hashes(hashes)

        print(f"✅ Embedded {len(new_docs)} new docs and updated FAISS index")


