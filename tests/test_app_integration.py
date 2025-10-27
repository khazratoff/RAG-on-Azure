import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_and_rag():
    if "backend.app" in sys.modules:
        del sys.modules["backend.app"]

    with patch("azure.storage.blob.BlobServiceClient") as BlobCls, \
         patch("azure.search.documents.SearchClient") as SearchCls, \
         patch("langchain_openai.embeddings.OpenAIEmbeddings") as EmbCls, \
         patch("langchain_openai.ChatOpenAI") as ChatCls:

        BlobCls.from_connection_string.return_value = MagicMock()
        BlobCls.return_value = MagicMock()
        SearchCls.return_value = MagicMock()
        EmbCls.return_value = MagicMock()
        ChatCls.return_value = MagicMock()

        import backend.app as app_module
        return app_module.app, app_module.rag


@pytest.fixture()
def client(app_and_rag):
    app, _ = app_and_rag
    return TestClient(app)


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("message").startswith("Welcome")


def test_stream_endpoint_streams_tokens(client, app_and_rag):
    _, rag = app_and_rag

    def fake_run(question):
        for t in ["x", "y", "z"]:
            yield t

    with patch.object(rag, "run", side_effect=fake_run):
        with client.stream("GET", "/stream", params={"question": "hi"}) as resp:
            assert resp.status_code == 200
            text = b"".join(resp.iter_raw()).decode("utf-8")
            assert "x" in text and "y" in text and "z" in text
