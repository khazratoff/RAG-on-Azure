import asyncio
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from AzureRagPipeline import AzureSearchRagPipeline
from redis_utils import load_chat_history, save_chat_history

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from azure.monitor.opentelemetry import configure_azure_monitor


load_dotenv()

# ---- FastAPI app setup ----
app = FastAPI(title="RAG Assistant")

# ---- Configuration ----
rag_cfg = {
    "azure": {
        "connection_string": os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        "blob_account_url": os.getenv("AZURE_BLOB_ACCOUNT_URL"),
        "raw_container": os.getenv("AZURE_RAW_CONTAINER", "raw-docs"),
        "vector_container": os.getenv("AZURE_VECTOR_CONTAINER", "vector-indexes"),
    },
    "search": {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX"),
        "api_key": os.getenv("AZURE_SEARCH_API_KEY"),
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    },
    "fastapi": {
        "allowed_origin": os.getenv("ALLOWED_ORIGIN", "*"),
    },
}

# ---- Enable CORS ----
app.add_middleware(
    CORSMiddleware,
    # allow_origins=[rag_cfg["fastapi"]["allowed_origin"]],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Initialize the RAG pipeline ----
rag = AzureSearchRagPipeline(cfg=rag_cfg)



# Initialize OpenTelemetry tracing
def setup_tracing():
    """Setup OpenTelemetry tracing with Azure Monitor exporter"""

    trace_provider = TracerProvider()
    FastAPIInstrumentor.instrument_app(app)

    # Add Azure Monitor exporter if connection string is available
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if connection_string:
        configure_azure_monitor()
        azure_exporter = AzureMonitorTraceExporter.from_connection_string(connection_string)
        trace_provider.add_span_processor(BatchSpanProcessor(azure_exporter))
        logging.info("OpenTelemetry tracing configured with Azure Monitor")
    else:
        # Fallback to console exporter for local development
        console_exporter = ConsoleSpanExporter()
        trace_provider.add_span_processor(BatchSpanProcessor(console_exporter))
        logging.info("OpenTelemetry tracing configured with console exporter (no Azure Monitor connection string found)")
    
    trace.set_tracer_provider(trace_provider)

setup_tracing()

# tracer = trace.get_tracer(__name__)


@app.get("/")
async def root():
    return {"message": "Employee Assistant API is running."}


@app.get("/stream")
async def stream(question: str, session_id: str):
    """
    Streams responses from the RAG pipeline with full tracing to Application Insights.
    """

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    history = load_chat_history(session_id)
    history.append({"role": "user", "content": question})
    # with tracer.start_as_current_span("fastapi_stream_request") as span:
    #     span.set_attribute("user.session_id", session_id)
    #     span.set_attribute("rag.user_question", question)
    #     span.add_event("streaming_started")
    async def event_generator():
            # Start a trace span for this request
                partial_response = ""
                try:
                    for chunk in rag.run(user_input=question, chat_history=history):
                        partial_response += chunk
                        yield chunk
                        await asyncio.sleep(0.05)

                    history.append({"role": "assistant", "content": partial_response})
                    save_chat_history(session_id, history)

                    # span.add_event("streaming_completed")
                    # span.set_attribute("response_length", len(partial_response))

                except Exception as e:
                    # span.record_exception(e)
                    yield f"data: Error: {str(e)}\n\n"

    return EventSourceResponse(event_generator())
