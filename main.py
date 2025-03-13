# github repo: https://github.com/g-pzusu/gq_pxy

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from typing import Dict, Any
import os
from dotenv import load_dotenv
from groq import AsyncGroq

# Load environment variables
load_dotenv()

app = FastAPI(
    docs_url=None,    # Disable docs (Swagger UI)
    redoc_url=None    # Disable redoc
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add API key security scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

# API key validation function
async def get_api_key(api_key: str = Depends(api_key_header)):
    # Remove 'Bearer ' prefix if present
    if api_key.startswith('Bearer '):
        api_key = api_key[7:]
    
    if api_key != "az-intital-key":
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key

# Get GROQ API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize AsyncGroq client
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

async def stream_groq_response(data: Dict[Any, Any]):
    try:
        completion = await groq_client.chat.completions.create(
            model=data.get("model", "deepseek-r1-distill-llama-70b"),
            messages=data.get("messages", []),
            temperature=data.get("temperature", 0.6),
            max_completion_tokens=data.get("max_tokens", 1024),
            top_p=data.get("top_p", 0.95),
            stream=data.get("stream", True),
            reasoning_format=data.get("reasoning_format", "raw")
        )
        
        async for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield f"data: {chunk.model_dump_json()}\n\n"
        
        yield "data: [DONE]\n\n"
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.post("/openai/v1/chat/completions")
async def chat_completions(request: Dict[Any, Any], api_key: str = Depends(get_api_key)):
    # Ensure stream is set to True as we're creating a streaming endpoint
    request["stream"] = True
    
    return StreamingResponse(
        stream_groq_response(request),
        media_type="text/event-stream"
    )

# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await groq_client.aclose()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8006))
    uvicorn.run(app, host="0.0.0.0", port=port)
