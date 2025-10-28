import os
import json
from typing import List, Dict

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class AzureSearchRagPipeline:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
# NOT for Prod---------------------
        # self.default_credential = DefaultAzureCredential()

        # self.blob_service = BlobServiceClient(
        #     account_url=self.cfg["azure"]["blob_account_url"],
        #     credential=self.default_credential
        # )
        self.blob_service = BlobServiceClient.from_connection_string(cfg["azure"]["connection_string"])
# NOT for Prod--------------------------
        # self.embeddings = AzureOpenAIEmbeddings(
        #     api_key=self.cfg["openai"]["api_key"],
        #     azure_endpoint=self.cfg["openai"]["endpoint"],
        #     azure_deployment=self.cfg["openai"]["deployment"],
        #     api_version=self.cfg["openai"]["api_version"]
        # )
        self.embeddings = OpenAIEmbeddings(
            api_key=self.cfg["openai"]["api_key"],
            model=self.cfg['openai']['embedding_model'],
        )
        
# NOT for Prod
        # self.llm = AzureChatOpenAI(
        #     api_key=self.cfg["openai"]["api_key"],
        #     azure_endpoint=self.cfg["openai"]["endpoint"],
        #     model=self.cfg["openai"]["chat_model"],
        #     api_version=self.cfg["openai"]["api_version"]
        # )
        self.llm = ChatOpenAI(
            streaming=True,
            api_key=self.cfg["openai"]["api_key"],
            model=self.cfg['openai']['chat_model'],
        )

        self.search_client = SearchClient(
            endpoint=self.cfg["search"]["endpoint"],
            index_name=self.cfg["search"]["index_name"],
            credential=AzureKeyCredential(self.cfg["search"]["api_key"])
        )

        self.vector_field = "content_vector"




    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k documents from Azure Search using vector similarity.
        """
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
                "metadata":r["metadata"],
                # "metadata": r.get("metadata", ""),
                "chunk_index": r.get("chunk_index", -1)
            })

        return docs

#---------------Adding chat history as a context----------

    def refine_query_with_history(self, user_input: str, history: str, docs: List[Dict]) -> str:
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
        return reformulated.content.strip()


    def generate_answer_with_history(self, refined_query: str, retrieved_docs: List[Dict], history: str):
        context = "\n".join([f"- {d['content']} (Doc: {d['metadata']})" for d in retrieved_docs])
        template = """
        You are a helpful assistant using Retrieval-Augmented Generation (RAG).
        Incorporate chat history to provide a coherent, context-aware answer.
        In addition to context there is a metadata which is basically name of the document. IF information you are providing stated in specific document make sure to include correct document name in your response otherwise NO need to include document name.
        If information is missing, be honest.

        Chat History:
        {history}

        Question: {query}
        Context:
        {context}

        Final Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["history", "query", "context"])

        for token in self.llm.stream(prompt.format(history=history, query=refined_query, context=context)):
            yield token.content

        
    def run(self, user_input: str, chat_history):
        """
        Executes full RAG flow with conversational history.
        """

        history_context = "\n".join(
            [f"{h['role'].capitalize()}: {h['content']}" for h in chat_history[-5:]]
        )  

        docs = self.retrieve(user_input)

        refined_query = self.refine_query_with_history(user_input, history_context, docs)

        stream = self.generate_answer_with_history(refined_query, docs, history_context)
        for chunk in stream:
            yield chunk  

