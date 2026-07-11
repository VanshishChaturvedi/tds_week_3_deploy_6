from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import numpy as np
from typing import List

app = FastAPI()

class RetrievalPayload(BaseModel):
    query_id: str
    query: str
    candidates: List[str]

def get_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    # Standard OpenAI Embeddings endpoint
    url = "https://api.openai.com/v1/embeddings"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "input": texts,
        "model": "text-embedding-3-small"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Embedding API error: {response.text}")
        
    data = response.json()["data"]
    # Ensure the returned embeddings are in the exact order requested
    sorted_data = sorted(data, key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    # Convert lists to NumPy arrays for mathematical operations
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    # Cosine similarity formula: dot product / (norm(v1) * norm(v2))
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

@app.post("/")
async def process_retrieval(payload: RetrievalPayload):
    # Fetch the OpenAI API key (ensure you add this to Render!)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set.")

    try:
        # Optimization: Combine query and all candidates into a single batch request
        all_texts = [payload.query] + payload.candidates
        all_embeddings = get_embeddings(all_texts, api_key)
        
        # The first embedding is the query, the rest are candidates
        query_embedding = all_embeddings[0]
        candidate_embeddings = all_embeddings[1:]
        
        # Calculate cosine similarity between the query and each candidate
        similarities = []
        for idx, cand_emb in enumerate(candidate_embeddings):
            sim_score = cosine_similarity(query_embedding, cand_emb)
            similarities.append((idx, sim_score))
            
        # Sort by similarity score in descending order (highest score first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the original indices of the top 3 highest scoring candidates
        top_3_indices = [sim[0] for sim in similarities[:3]]
        
        # Return strict expected JSON structure
        return {"ranking": top_3_indices}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
