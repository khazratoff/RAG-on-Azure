import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
# from hydra import initialize, compose
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

class RagPipeline:
    def __init__(self, config):
        self.config = config
        self.vectorstore_path = config.vectorstore.path
        self.account_url = config.azure.account_url
        self.default_credential = DefaultAzureCredential()
        self.blob_service = BlobServiceClient(account_url=self.account_url, credential=self.default_credential)
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=config.azure.api_key,
            azure_endpoint=config.azure.endpoint,
            azure_deployment=config.azure.deployment.embedding,
            api_version="2023-05-15"
        )
        self.llm = AzureChatOpenAI(
            api_key=config.azure.api_key,
            azure_endpoint=config.azure.endpoint,
            model=config.azure.deployment.chat,
            api_version="2023-05-15"
        )
        print("ℹ️ Retrieving docs from Azure Blob container...")
        self.load_vectorsore(self.config.azure.storage.container.vector, self.vectorstore_path)

    def load_vectorsore(self,container_name: str, local_dir: str):
        """Download all files from a blob container into a local directory."""

        print("ℹ️ Loading existing FAISS index")
        os.makedirs(local_dir, exist_ok=True)
        container_client = self.blob_service.get_container_client(container_name)
        for blob in container_client.list_blobs():
            blob_client = container_client.get_blob_client(blob.name)
            local_path = os.path.join(local_dir, blob.name)
            with open(local_path, "wb") as f:
                f.write(blob_client.download_blob().readall())

    def retrieve(self, query: str):

        faiss_store = FAISS.load_local(
            self.vectorstore_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        docs = faiss_store.similarity_search(query, k=5)
        return docs

    def generate_query(self, user_input: str, context_docs: list):
        template = """
        You are a smart assistant. Rewrite the following user question to make it clearer and more precise 
        for retrieval. Use the given context if relevant.

        Context: {context}
        User Question: {question}
        Reformulated Query:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        context_text = " ".join([doc.page_content for doc in context_docs])
        formatted_prompt = prompt.format(context=context_text, question=user_input)

        response = self.llm.invoke(formatted_prompt)
        return response.content.strip()

    def generate_answer(self, query: str, retrieved_docs: list):
        template = """
        You are an intelligent assistant using RAG.
        Answer the user’s query based on the retrieved context. 
        If context is insufficient, say so transparently.

        Query: {query}
        Context: {context}
        Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["query", "context"])
        context_text = " ".join([doc.page_content for doc in retrieved_docs])
        formatted_prompt = prompt.format(query=query, context=context_text)

        response = self.llm.invoke(formatted_prompt)
        return response.content.strip()

    def run(self, user_input: str):
        docs = self.retrieve(user_input)

        refined_query = self.generate_query(user_input, docs)

        answer = self.generate_answer(refined_query, docs)
        return answer


# def main():
#     with initialize(config_path="../configs",version_base="1.1"):
#         cfg = compose(config_name="config")

#     rag = RagPipeline(config=cfg)
#     print(rag.run("when the company founded?"))


# if __name__ == "__main__":
#     main()
