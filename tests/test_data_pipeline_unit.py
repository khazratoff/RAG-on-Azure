import json
from unittest.mock import MagicMock

import pytest

from DataPipeline.AzureDataPipeline import AzureSearchDataPipeline


def make_cfg():
    return {
        "azure": {
            "blob_account_url": "https://example.blob.core.windows.net",
            "connection_string": "UseDevelopmentStorage=true;",
            "raw_container": "raw-docs",
            "vector_container": "vector-indexes",
        },
        "search": {
            "endpoint": "https://example.search.windows.net",
            "index_name": "test-index",
            "api_key": "test-key",
        },
        "openai": {
            "api_key": "test",
            "embedding_model": "text-embedding-3-small",
        },
        "chunker": {"chunk_size": 10, "chunk_overlap": 0},
        "batch": {"embed_batch_size": 2, "upsert_batch_size": 2},
    }


def test_compute_hash():
    pipeline = AzureSearchDataPipeline(make_cfg())
    h1 = pipeline.compute_hash(b"abc")
    h2 = pipeline.compute_hash(b"abc")
    h3 = pipeline.compute_hash(b"abcd")
    assert h1 == h2 and h1 != h3


def test_build_documents_for_search(mock_openai_embeddings, mock_blob_service, mock_search_client):
    pipeline = AzureSearchDataPipeline(make_cfg())
    chunks = ["hello", "world"]
    metas = [
        {"metadata": "file1.txt", "chunk_index": 0, "hash": "h1"},
        {"metadata": "file1.txt", "chunk_index": 1, "hash": "h1"},
    ]
    vectors = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

    docs = pipeline.build_documents_for_search(chunks, metas, vectors)

    assert len(docs) == 2
    assert docs[0]["id"] == "h1_0"
    assert docs[1]["content"] == "world"
    assert "content_vector" in docs[0]


def test_scan_for_new_chunks_text_only(mock_openai_embeddings, mock_blob_service, mock_search_client, monkeypatch):
    cfg = make_cfg()
    pipeline = AzureSearchDataPipeline(cfg)

    raw_container = pipeline.blob_service.get_container_client.return_value

    blob1 = MagicMock()
    blob1.name = "doc1.txt"

    blob_client = MagicMock()
    blob_client.download_blob.return_value.readall.return_value = b"hello\nworld"

    raw_container.list_blobs.return_value = [blob1]
    raw_container.get_blob_client.return_value = blob_client

    # No stored hashes
    pipeline.load_hashes = lambda: set()

    chunks, metas, new_hashes = pipeline.scan_for_new_chunks()

    assert chunks and metas
    assert len(chunks) == len(metas)
    assert any(m["chunk_index"] == 0 for m in metas)
    assert len(new_hashes) == 1


def test_embed_batches(mock_openai_embeddings, mock_blob_service, mock_search_client):
    pipeline = AzureSearchDataPipeline(make_cfg())
    texts = ["a", "b", "c"]
    vecs = pipeline.embed_batches(texts)
    assert len(vecs) == len(texts)


def test_upsert_batches_creates_index_if_missing(mock_openai_embeddings, mock_blob_service, mock_search_client, monkeypatch):
    pipeline = AzureSearchDataPipeline(make_cfg())

    # Simulate ResourceNotFoundError on first upload, then succeed
    from azure.core.exceptions import ResourceNotFoundError

    called = {"attempt": 0}

    def upload_documents(documents):
        if called["attempt"] == 0:
            called["attempt"] += 1
            raise ResourceNotFoundError("not found")
        return [MagicMock(succeeded=True)]

    pipeline.search_client.upload_documents.side_effect = upload_documents
    pipeline._create_index_if_missing = MagicMock()

    docs = [
        {"id": "1", "content": "a", "metadata": "f", "chunk_index": 0, "hash": "h", "content_vector": [0.1, 0.2, 0.3]},
        {"id": "2", "content": "b", "metadata": "f", "chunk_index": 1, "hash": "h", "content_vector": [0.1, 0.2, 0.3]},
    ]

    pipeline.upsert_batches(docs)

    assert pipeline._create_index_if_missing.called
    assert called["attempt"] == 1
