import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is missing!")

# Initialize Clients
app = FastAPI(
    title="Production AI Backend Engine",
    description="High-speed AI microservice built with FastAPI & Groq LPU engine",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------
# Request & Response Schemas (Pydantic Validation)
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    prompt: str = Field(..., example="Explain vector embeddings in 2 sentences.")
    model: str = Field(default="llama-3.3-70b-versatile", example="llama-3.3-70b-versatile")

class CodeAnalysisRequest(BaseModel):
    code_snippet: str = Field(..., example="def add(a, b): return a + b")

class CodeAnalysisResponse(BaseModel):
    language: str
    complexity_score: str
    bugs_found: list[str]
    optimized_code: str


# ---------------------------------------------------------
# Endpoint 1: Real-time SSE Streaming Engine
# ---------------------------------------------------------
@app.post("/api/v1/stream")
async def stream_ai_response(request: ChatRequest):
    """
    Streams LLM token outputs in real-time using Server-Sent Events (SSE).
    """
    def event_generator():
        try:
            stream = client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": "You are a concise, world-class AI system assistant."},
                    {"role": "user", "content": request.prompt}
                ],
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {content}\n\n"
        except Exception as e:
            yield f"data: Error generating response: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------
# Endpoint 2: Structured Data Extraction (JSON Schema)
# ---------------------------------------------------------
@app.post("/api/v1/analyze-code", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    """
    Enforces guaranteed structured output from LLMs using JSON Mode + Pydantic.
    """
    system_prompt = f"""
    You are an expert code reviewer. Analyze the given code and return ONLY a raw JSON object matching this strict schema:
    {json.dumps(CodeAnalysisResponse.model_json_schema())}
    Do not include markdown formatting or commentary outside the JSON object.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.code_snippet}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_output = completion.choices[0].message.content
        parsed_json = json.loads(raw_output)
        
        # Pydantic validates raw LLM output against our schema
        return CodeAnalysisResponse(**parsed_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Parsing failed: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "online", "model_engine": "Groq LPU"}