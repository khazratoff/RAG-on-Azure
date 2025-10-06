from fastapi import FastAPI
from pydantic import BaseModel
from rag import RagPipeline  

from hydra import initialize, compose

app = FastAPI()

with initialize(config_path="../configs", version_base="1.1"):
    cfg = compose(config_name="config")

rag = RagPipeline(config=cfg)


class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    answer = rag.run(query.question)
    return {"answer": answer}
