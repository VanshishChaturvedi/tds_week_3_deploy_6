from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import math
from typing import List

app = FastAPI()

# Pydantic model for the incoming payload
class RetrievalPayload(BaseModel):
    query_id: str
    query: str
    candidates: List[str]

# Pure Python Cosine Similarity (No Numpy required!)
def cos_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

@app.post("/")
async def process_retrieval(payload: RetrievalPayload):
    # Using your existing key that is already set up in Render
    api_key = os.getenv("GEMINI_API_KEY") 
    if not api_key:
        raise HTTPException(status_code=500, detail="API key environment variable not set.")

    # CRITICAL FIX: Routing to the AIPipe proxy's OpenAI embeddings endpoint
    url = "https://aipipe.org/openai/v1/embeddings"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Batching the query and all candidates into a single request
    data = {
        "model": "text-embedding-3-small",
        "input": [payload.query] + payload.candidates
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        # Extract the embeddings exactly as the resource demonstrated
        vecs = [d["embedding"] for d in response.json()["data"]]
        
        # The first vector is the query, the rest are candidates
        q_vec = vecs[0]
        cand_vecs = vecs[1:]
        
        # Calculate cosine similarity and sort indices in descending order
        scored_indices = sorted(range(len(cand_vecs)), key=lambda i: cos_sim(q_vec, cand_vecs[i]), reverse=True)
        
        # Return the exact indices of the top 3 highest scoring candidates
        return {"ranking": scored_indices[:3]}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API Request failed: {str(e)}")
