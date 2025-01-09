from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import PlainTextResponse
import time
from fastapi.responses import JSONResponse, PlainTextResponse
import redis
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from pydantic import BaseModel
import logging
import psutil
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define metrics (only once)
REQUEST_COUNT = Counter("apiserver_request_total", "Total number of requests")
REQUEST_LATENCY = Histogram("apiserver_request_latency_seconds", "Latency of requests in seconds")
ERROR_COUNT = Counter("apiserver_request_errors_total", "Total number of errors")
ACTIVE_REQUESTS = Gauge("apiserver_active_requests", "Number of active requests")
QUEUE_DURATION_BUCKET = Histogram("workqueue_queue_duration_seconds_bucket", "Time tasks spend in queue", buckets=[0.1, 0.5, 1, 2, 5, 10])
API_REQUEST_DURATION_BUCKET = Histogram("apiserver_request_duration_seconds_bucket", "Duration of API server requests", buckets=[0.1, 0.5, 1, 2, 5, 10])

# CPU and memory metrics
CPU_USAGE = Gauge("node_cpu_usage_percent", "Total CPU usage percentage")
MEMORY_TOTAL = Gauge("node_memory_total_bytes", "Total memory in bytes")
MEMORY_FREE = Gauge("node_memory_free_bytes", "Free memory in bytes")
MEMORY_CACHED = Gauge("node_memory_cached_bytes", "Cached memory in bytes")
MEMORY_BUFFERS = Gauge("node_memory_buffers_bytes", "Memory used by buffers in bytes")
MEMORY_RECLAIMABLE = Gauge("node_memory_reclaimable_bytes", "Reclaimable memory in bytes")
CPU_USER_SECONDS_TOTAL = Gauge("app_cpu_user_seconds_total", "Total seconds of user CPU usage")
CPU_SYSTEM_SECONDS_TOTAL = Gauge("app_cpu_system_seconds_total", "Total seconds of system CPU usage")
CPU_IDLE_SECONDS_TOTAL = Gauge("app_cpu_idle_seconds_total", "Total seconds of idle CPU time")
CPU_IOWAIT_SECONDS_TOTAL = Gauge("app_cpu_iowait_seconds_total", "Total seconds of I/O wait CPU time")
CPU_OTHER_SECONDS_TOTAL = Gauge("app_cpu_other_seconds_total", "Total seconds of other CPU time")

API_PROBE_DURATION_SECONDS = Histogram("api_probe_duration_seconds", "Duration of API request processing time")

# Initialize the FastAPI app
app = FastAPI()

# Function to update general metrics
def update_metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))

    # Memory Usage
    virtual_memory = psutil.virtual_memory()
    MEMORY_TOTAL.set(virtual_memory.total)
    MEMORY_FREE.set(virtual_memory.free)
    MEMORY_CACHED.set(getattr(virtual_memory, 'cached', 0))
    MEMORY_BUFFERS.set(getattr(virtual_memory, 'buffers', 0))
    MEMORY_RECLAIMABLE.set(getattr(virtual_memory, 'shared', 0))

# Function to update CPU mode metrics
def update_cpu_metrics():
    cpu_times = psutil.cpu_times()

    # Set metrics for known modes
    CPU_USER_SECONDS_TOTAL.set(cpu_times.user)
    CPU_SYSTEM_SECONDS_TOTAL.set(cpu_times.system)
    CPU_IDLE_SECONDS_TOTAL.set(cpu_times.idle)
    CPU_IOWAIT_SECONDS_TOTAL.set(getattr(cpu_times, 'iowait', 0))

    # Calculate "other" CPU time by subtracting known modes from total
    total_known_time = cpu_times.user + cpu_times.system + cpu_times.idle + getattr(cpu_times, 'iowait', 0)
    CPU_OTHER_SECONDS_TOTAL.set(total_known_time)

@app.on_event("startup")
async def setup_metrics():
    # Initialize metrics at startup
    update_metrics()
    update_cpu_metrics()

# Endpoint for Prometheus metrics
@app.get("/metrics")
async def metrics():
    update_metrics()
    update_cpu_metrics()
    return PlainTextResponse(generate_latest())

# Middleware to track metrics for each request
@app.middleware("http")
async def track_request_duration(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time 
    latency = time.time() - start_time
    API_REQUEST_DURATION_BUCKET.observe(latency)
    API_PROBE_DURATION_SECONDS.observe(duration)  # Record the duration
    return response


async def track_metrics(request: Request, call_next):
    REQUEST_COUNT.inc()  # Increment total request count
    ACTIVE_REQUESTS.inc()  # Increment active requests count

    # Start time to measure request latency
    start_time = time.time()
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        ERROR_COUNT.inc()  # Increment error count on exception
        raise e
    finally:
        # Calculate latency and update metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        ACTIVE_REQUESTS.dec()  # Decrement active requests count after completion

# Define request model
class ChatRequest(BaseModel):
    question: str

# Connect to Redis
cache = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Set up the template and LLM model
template = """
Conversation history: {context}

Based on this conversation history, respond only to the following question. 

Question: {question}
"""

template_new = """
{question}
"""

model = OllamaLLM(model="llama3.2:1b")
prompt = ChatPromptTemplate.from_template(template)
prompt_new = ChatPromptTemplate.from_template(template_new)
chain = prompt | model
chain_new = prompt_new | model

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    REQUEST_COUNT.inc()  # Increment the request count
    question = request.question

    start_time = time.time()  # Start timing the request

    try:
        # Check if the question exists in the cache
        if cache.exists(question):
            answer = cache.get(question)
            logger.info(f"Cache hit for question: {question}")
            response_time = time.time() - start_time
            REQUEST_LATENCY.observe(response_time)  # Observe latency for cached response
            return {"answer": answer}

        # Load and trim the conversation context
        context = cache.get("context") or ""
        trimmed_context = context
        if context == "":
            result = chain_new.invoke({"question": question})
        else:
            context_lines = context.strip().split("\n")

            # Keep only the last 10 lines (5 exchanges)
            if len(context_lines) > 10:
                context_lines = context_lines[-10:]

            # Re-assemble the trimmed context
            trimmed_context = "\n".join(context_lines)

            # Generate response using the model with trimmed context
            result = chain.invoke({"context": trimmed_context, "question": question})
        
        # Update cache with the result and new context
        cache.setex(question, 600, result)  # Cache the result for 10 minutes
        new_context = trimmed_context + f"\nUser: {question}\nAI: {result}"
        cache.setex("context", 3600, new_context)  # Cache the updated context for 1 hour

        logger.info(f"Generated answer: {result}")

        # Calculate and log response time
        response_time = time.time() - start_time
        REQUEST_LATENCY.observe(response_time)  # Observe latency for new response
        logger.info(f"Response time: {response_time:.2f} seconds")

        return {"answer": result}

    except Exception as e:
        response_time = time.time() - start_time
        REQUEST_LATENCY.observe(response_time)  # Observe latency for errors
        logger.error(f"Error processing question: {question} - {str(e)} (Response time: {response_time:.2f} seconds)")
        return JSONResponse(status_code=500, content={"error": "An error occurred"})