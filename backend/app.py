import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from redis_utils import load_chat_history, save_chat_history


load_dotenv()

from AzureRagPipeline import AzureSearchRagPipeline

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


rag_cfg = {
    "azure": {
        "blob_account_url": os.getenv("AZURE_BLOB_ACCOUNT_URL"),
        "raw_container": os.getenv("AZURE_RAW_CONTAINER", "raw-docs"),
        "vector_container": os.getenv("AZURE_VECTOR_CONTAINER", "vector-indexes"),
        "connection_string": os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    },
    "search": {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX"),
        "api_key": os.getenv("AZURE_SEARCH_API_KEY",),
    },
    "openai": {
        # "endpoint": os.getenv("OPENAI_AZURE_ENDPOINT"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        # "deployment": os.getenv("OPENAI_DEPLOYMENT"),
        # "api_version": os.getenv("OPENAI_API_VERSION", "2023-05-15"),
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    },
    "fastapi":{
        "allowed_origin":os.getenv("ALLOWED_ORIGIN")
    }
    }

app.add_middleware(
    CORSMiddleware,
    # allow_origins=[rag_cfg["fastapi"]["allowed_origin"]], 
    allow_origins=["*"], # for local development 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


rag = AzureSearchRagPipeline(cfg=rag_cfg)


@app.get("/")
def root():
    return {"message": "Welcome, this is Employee Assistant!"}



@app.get("/stream")
async def stream(question: str, session_id: str):

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    history = load_chat_history(session_id)

    history.append({"role": "user", "content": question})

    async def event_generator():
        partial_response = ""
        try:
            for chunk in rag.run(question, chat_history=history):
                partial_response += chunk
                yield chunk
                await asyncio.sleep(0.1)

            history.append({"role": "assistant", "content": partial_response})
            save_chat_history(session_id, history)

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return EventSourceResponse(event_generator())