from unittest.mock import MagicMock

import pytest

from backend.AzureRagPipeline import AzureSearchRagPipeline


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
            "chat_model": "gpt-4",
        },
        "fastapi": {"allowed_origin": "*"},
    }


def test_retrieve_uses_vector_query(mock_openai_embeddings, mock_search_client, mock_blob_service):
    pipeline = AzureSearchRagPipeline(make_cfg())

    # Configure search results mock
    result_item = {"content": "A", "metadata": "doc.txt", "chunk_index": 0}
    pipeline.search_client.search.return_value = [result_item]

    docs = pipeline.retrieve("hello", k=1)

    pipeline.embeddings.embed_query.assert_called_once()
    pipeline.search_client.search.assert_called_once()
    assert docs == [result_item]


def test_refine_and_generate_with_history_streaming(monkeypatch, mock_openai_embeddings, mock_search_client, mock_blob_service):
    pipeline = AzureSearchRagPipeline(make_cfg())

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="refined question")

    def fake_stream(_):
        for t in ["hello ", "world"]:
            yield MagicMock(content=t)

    fake_llm.stream.side_effect = fake_stream

    pipeline.llm = fake_llm

    docs = [{"content": "c1", "metadata": "doc1", "chunk_index": 0}]

    refined = pipeline.refine_query_with_history("q", "User: hi", docs)
    assert refined == "refined question"

    chunks = list(pipeline.generate_answer_with_history(refined, docs, "User: hi"))
    assert "".join(chunks) == "hello world"


def test_run_updates_chat_history(monkeypatch, mock_openai_embeddings, mock_search_client, mock_blob_service):
    pipeline = AzureSearchRagPipeline(make_cfg())

    pipeline.retrieve = lambda q: [{"content": "c1", "metadata": "doc1", "chunk_index": 0}]
    pipeline.refine_query_with_history = lambda q, h, d: "rq"

    def fake_stream(_rq, _docs, _h):
        for t in ["a", "b"]:
            yield t

    pipeline.generate_answer_with_history = fake_stream

    chunks = list(pipeline.run("hello"))
    assert chunks == ["a", "b"]
    # chat_history should have user and assistant entries
    assert len(pipeline.chat_history) == 2
    assert pipeline.chat_history[0]["role"] == "user"
    assert pipeline.chat_history[1]["role"] == "assistant"
