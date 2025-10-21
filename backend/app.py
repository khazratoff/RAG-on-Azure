import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv


load_dotenv()

from AzureRagPipeline import AzureSearchRagPipeline

app = FastAPI()



# API_KEY = os.getenv("FASTAPI_KEY")
# API_KEY_NAME = "PASS-KEY"

# def verify_api_key(request: Request):
#     api_key = request.headers.get(API_KEY_NAME)
#     if api_key != API_KEY:
#         raise HTTPException(status_code=401, detail="Invalid or missing API Key")

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
    allow_origins=[rag_cfg["fastapi"]["allowed_origin"]], 
    # allow_origins=["*"], for local testing
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

rag = AzureSearchRagPipeline(cfg=rag_cfg)

# @app.post("/chat")
# def chat_endpoint(request_body: QueryRequest, _: None = Depends(verify_api_key)):
#     response = rag.run(request_body.query)
#     return {"response": response}

@app.get("/")
def root():
    return {"message": "Welcome, this is Employee Assistant!"}



@app.get("/stream")
async def stream(question: str):
    async def event_generator():
        try:
            for chunk in rag.run(question):
                yield chunk
                await asyncio.sleep(0.1)  
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return EventSourceResponse(event_generator())