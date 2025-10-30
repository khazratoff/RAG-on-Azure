import os
import json
from typing import List, Dict
from contextlib import contextmanager
import time

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

#Tracing
from opentelemetry import trace


class AzureSearchRagPipeline:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

        # --- Azure Blob ---
        self.blob_service = BlobServiceClient.from_connection_string(cfg["azure"]["connection_string"])

        # --- Embeddings & LLM ---
        self.embeddings = OpenAIEmbeddings(
            api_key=self.cfg["openai"]["api_key"],
            model=self.cfg['openai']['embedding_model'],
        )

        self.llm = ChatOpenAI(
            streaming=True,
            api_key=self.cfg["openai"]["api_key"],
            model=self.cfg['openai']['chat_model'],
        )

        # --- Azure Cognitive Search ---
        self.search_client = SearchClient(
            endpoint=self.cfg["search"]["endpoint"],
            index_name=self.cfg["search"]["index_name"],
            credential=AzureKeyCredential(self.cfg["search"]["api_key"])
        )

        self.vector_field = "content_vector"

        #Trace
        self.tracer = trace.get_tracer(__name__)


    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k documents from Azure Search using vector similarity."""
        with self.tracer.start_as_current_span("retrieve_documents") as span:
            start = time.time()
            span.set_attribute("query_text", query)
            print(f"🔍 Retrieving top {k} matches for: {query}")

            query_vector = self.embeddings.embed_query(query)
            results = self.search_client.search(
                vector_queries=[
                    {
                        "kind": "vector",
                        "vector": query_vector,
                        "fields": self.vector_field,
                        "k": k,
                        "exhaustive": True
                    }
                ],
                select=["id", "content", "metadata", "chunk_index"]
            )

            docs = []
            for r in results:
                docs.append({
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "chunk_index": r.get("chunk_index", -1)
                })

            span.set_attribute("num_results", len(docs))
            span.set_attribute("latency_ms", (time.time() - start) * 1000)
            return docs


    def refine_query_with_history(self, user_input: str, history: str, docs: List[Dict]) -> str:
        """Use LLM to reformulate question using chat context."""
        with self.tracer.start_as_current_span("refine_query") as span:
            span.set_attribute("user_input", user_input)

            context = "\n".join([f"- {d['content']} (Doc: {d['metadata']})" for d in docs])
            template = """
            You are an intelligent assistant that reformulates user questions for better retrieval using prior chat context.
            Chat History:
            {history}
            Current Context:
            {context}
            User Query:
            {query}

            Reformulated Query:
            """
            prompt = PromptTemplate(template=template, input_variables=["history", "context", "query"])
            reformulated = self.llm.invoke(prompt.format(history=history, context=context, query=user_input))
            span.set_attribute("refined_query", reformulated.content.strip())
            span.set_attribute("history", history)
            return reformulated.content.strip()


    def generate_answer_with_history(self, refined_query: str, retrieved_docs: List[Dict], history: str):
        """Generate context-aware response from retrieved docs."""
        with self.tracer.start_as_current_span("generate_answer") as span:

            context = "\n".join([f"- {d['content']} (Doc: {d['metadata']})" for d in retrieved_docs])
            span.set_attribute("context", context)

            template = """
                You are a helpful assistant using Retrieval-Augmented Generation (RAG).
                Incorporate chat history to provide a coherent, context-aware answer.
                If information comes from a document, include its name.

                Chat History:
                {history}

                Question: {query}
                Context:
                {context}

                Final Answer:
                """
            prompt = PromptTemplate(template=template, input_variables=["history", "query", "context"])

            for token in self.llm.stream(prompt.format(history=history, query=refined_query, context=context)):
                # span.add_event("llm_stream_token", {"token": token.content})
                yield token.content


    def run(self, user_input: str, chat_history):
        """Full conversational RAG flow with tracing."""
        # with self.tracer.start_as_current_span("RAG_pipeline_run") as span:
        history_context = "\n".join(
                [f"{h['role'].capitalize()}: {h['content']}" for h in chat_history[-5:]]
            )
            # span.set_attribute("history context", history_context)

        docs = self.retrieve(user_input)
        refined_query = self.refine_query_with_history(user_input, history_context, docs)
        stream = self.generate_answer_with_history(refined_query, docs, history_context)
        for chunk in stream:
            yield chunk
