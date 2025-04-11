from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import asyncio
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from langchain_huggingface import HuggingFaceEmbeddings
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import pinecone  # Corrected import
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from dotenv import load_dotenv

# ----------------------------
# Create FastAPI app and rate limiter
# ----------------------------
app = FastAPI()

# Set up a rate limiter: 10 requests per minute per client IP
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Custom exception handler for rate limits
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

# ----------------------------
# Configure external API keys
# ----------------------------
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ----------------------------
# Define Pydantic model for request
# ----------------------------
class QueryRequest(BaseModel):
    query: str

# ----------------------------
# Global variables for models and vector store
# ----------------------------
embedding_model = None
vectorstore = None
gemini_model = None

# ----------------------------
# Startup event
# ----------------------------
@app.on_event("startup")
async def startup_event():
    global embedding_model, vectorstore, gemini_model

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")

    # Initialize Pinecone client
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])

    index_name = "blake"
    if index_name not in pinecone.list_indexes():
        raise RuntimeError(f"Pinecone index '{index_name}' does not exist.")

    # Connect to existing index using LangChain
    vectorstore = LC_Pinecone.from_existing_index(index_name, embedding_model)

    # Initialize Gemini
    gemini_model = genai.GenerativeModel("gemini-pro")

# ----------------------------
# Helper: similarity search
# ----------------------------
async def get_relevant_documents(query: str, k: int = 3) -> str:
    docs = await asyncio.to_thread(vectorstore.similarity_search, query, k)
    return "\n\n".join([doc.page_content for doc in docs])

# ----------------------------
# Helper: query Gemini with backoff
# ----------------------------
async def query_gemini(context: str, user_query: str) -> str:
    max_attempts = 5
    attempt = 0
    delay = 1

    prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's query.
    Remove any newline (\\n) and tab (\\t) characters in your answer.
    Context:
    {context}
    User Query:
    {user_query}
    """

    while attempt < max_attempts:
        try:
            await asyncio.sleep(0.5)
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except ResourceExhausted:
            attempt += 1
            await asyncio.sleep(delay)
            delay *= 2

    raise HTTPException(
        status_code=500,
        detail="Failed to generate content after multiple attempts due to quota/resource exhaustion."
    )

@app.get("/ping")
async def ping():
    return {"status": "ok"}

# ----------------------------
# /ask endpoint
# ----------------------------
@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, query_request: QueryRequest):
    user_query = query_request.query
    context = await get_relevant_documents(user_query)
    response_text = await query_gemini(context, user_query)
    return {"response": response_text}