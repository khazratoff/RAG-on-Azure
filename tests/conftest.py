import os
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("AZURE_SEARCH_API_KEY", "test-key")
os.environ.setdefault("AZURE_SEARCH_ENDPOINT", "https://example.search.windows.net")
os.environ.setdefault("AZURE_SEARCH_INDEX", "test-index")
os.environ.setdefault("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true;")
os.environ.setdefault("AZURE_BLOB_ACCOUNT_URL", "https://example.blob.core.windows.net")
os.environ.setdefault("ALLOWED_ORIGIN", "*")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("NO_NETWORK", "1")
    yield


@pytest.fixture(autouse=True)
def patch_sdk_clients():
    with patch("DataPipeline.AzureDataPipeline.BlobServiceClient") as BlobDP, \
         patch("DataPipeline.AzureDataPipeline.SearchClient") as SearchDP, \
         patch("DataPipeline.AzureDataPipeline.OpenAIEmbeddings") as EmbDP, \
         patch("backend.AzureRagPipeline.BlobServiceClient") as BlobRAG, \
         patch("backend.AzureRagPipeline.SearchClient") as SearchRAG, \
         patch("backend.AzureRagPipeline.OpenAIEmbeddings") as EmbRAG, \
         patch("backend.AzureRagPipeline.ChatOpenAI") as ChatRAG:

        blob_service = MagicMock()
        BlobDP.from_connection_string.return_value = blob_service
        BlobDP.return_value = blob_service
        BlobRAG.from_connection_string.return_value = blob_service
        BlobRAG.return_value = blob_service

        search_client = MagicMock()
        SearchDP.return_value = search_client
        SearchRAG.return_value = search_client

        emb_instance = MagicMock()
        emb_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        emb_instance.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        EmbDP.return_value = emb_instance
        EmbRAG.return_value = emb_instance

        ChatRAG.return_value = MagicMock()

        yield


@pytest.fixture()
def mock_openai_embeddings():
    with patch("langchain_openai.embeddings.OpenAIEmbeddings") as cls:
        instance = MagicMock()
        instance.embed_query.return_value = [0.1, 0.2, 0.3]
        instance.embed_documents.side_effect = lambda texts: [[0.1] * 3 for _ in texts]
        cls.return_value = instance
        yield instance


@pytest.fixture()
def mock_search_client():
    with patch("azure.search.documents.SearchClient") as cls:
        instance = MagicMock()
        instance.search.return_value = []
        instance.upload_documents.return_value = []
        cls.return_value = instance
        yield instance


@pytest.fixture()
def mock_blob_service():
    with patch("azure.storage.blob.BlobServiceClient") as cls:
        service = MagicMock()
        container = MagicMock()
        service.get_container_client.return_value = container
        cls.from_connection_string.return_value = service
        cls.return_value = service
        yield service
