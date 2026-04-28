from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="Spam Detection LLM Service")

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

class Query(BaseModel):
    prompt: str
    model: str = "qwen2.5:0.5b"

@app.get("/")
def home():
    return {"message": "LLM Spam Detection Service is running"}

@app.post("/generate")
def generate(query: Query):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": query.model,
                "prompt": query.prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return {
            "prompt": query.prompt,
            "response": result.get("response", ""),
            "model": query.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))