import os
import json
from typing import List, Dict

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate
from configs.config import rag_cfg


class AzureSearchRagPipeline:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

        self.default_credential = DefaultAzureCredential()

        self.blob_service = BlobServiceClient(
            account_url=self.cfg["azure"]["blob_account_url"],
            credential=self.default_credential
        )

        self.embeddings = AzureOpenAIEmbeddings(
            api_key=self.cfg["openai"]["api_key"],
            azure_endpoint=self.cfg["openai"]["endpoint"],
            azure_deployment=self.cfg["openai"]["deployment"],
            api_version=self.cfg["openai"]["api_version"]
        )

        self.llm = AzureChatOpenAI(
            api_key=self.cfg["openai"]["api_key"],
            azure_endpoint=self.cfg["openai"]["endpoint"],
            model=self.cfg["openai"]["chat_model"],
            api_version=self.cfg["openai"]["api_version"]
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


    def refine_query(self, user_input: str, docs: List[Dict]) -> str:
        """
        Use LLM to reformulate the user’s question for more accurate retrieval.
        """
        context=''
        for d in docs:
            context += f"- {d['content']}\n"
            context+=f"Document name: {d['metadata']}\n"
        # context = " ".join([d["content"],d["metadata"] for d in docs])
        template = """
        You are an intelligent assistant that reformulates user questions for better retrieval.
        Context: {context}
        User Query: {query}
        Reformulated Query:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "query"])
        reformulated = self.llm.invoke(prompt.format(context=context, query=user_input))
        return reformulated.content.strip()


    def generate_answer(self, refined_query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate a final answer using the LLM and retrieved context.
        """
        context=''
        for d in retrieved_docs:
            context += f"- {d['content']}\n"
            context+=f"Document name: {d['metadata']}\n"
        # context = "\n".join([f"- {d['content']}" for d in retrieved_docs])
        template = """
        You are a helpful assistant using Retrieval-Augmented Generation (RAG).
        Use the context to answer the question accurately.
        In addition to context there is a metadata which is basically name of the document. Make sure to include correct document name in your response.

        If information is missing, be honest.

        Question: {query}
        Context:
        {context}

        Final Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["query", "context"])
        response = self.llm.invoke(prompt.format(query=refined_query, context=context))
        return response.content.strip()


    def run(self, user_input: str) -> str:
        """
        Execute full RAG flow: retrieve → refine → generate answer.
        """
        # Step 1: Retrieve documents
        docs = self.retrieve(user_input)

        # Step 2: Refine query
        refined_query = self.refine_query(user_input, docs)

        # Step 3: Generate final answer
        answer = self.generate_answer(refined_query, docs)
        return answer



def main():
    
    rag_pipeline = AzureSearchRagPipeline(cfg=rag_cfg)
    
    user_query = "What is the most strict company rules?"
    answer = rag_pipeline.run(user_query)
    print(f"Answer: {answer}")
if __name__ == "__main__":
    main()