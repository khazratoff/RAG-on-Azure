from fastapi import FastAPI
from pydantic import BaseModel
from backend.AzureRagPipeline import AzureSearchRagPipeline
from configs.config import rag_cfg


app = FastAPI()


rag_pipeline = AzureSearchRagPipeline(cfg=rag_cfg)


class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    answer = rag_pipeline.run(query.question)
    return {"answer": answer}
