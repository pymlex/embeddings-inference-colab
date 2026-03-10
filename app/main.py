from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.model import Embedder
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.environ.get("MODEL_DIR", "mlsa-iai-msu-lab/sci-rus-tiny")
embedder = Embedder(MODEL_DIR)

def extract_texts(payload):
    if isinstance(payload, dict):
        if "inputs" in payload and payload["inputs"] is not None:
            return list(payload["inputs"])
        if "input" in payload and payload["input"] is not None:
            if isinstance(payload["input"], list):
                return list(payload["input"])
            return [payload["input"]]
        return None
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, str):
        return [payload]
    return None

def build_openai_response(embs):
    items = []
    for i, e in enumerate(embs):
        items.append({"object": "embedding", "embedding": e, "index": i})
    return {"object": "list", "data": items, "model": MODEL_DIR}

def handle_request_payload(payload):
    texts = extract_texts(payload)
    if texts is None or len(texts) == 0:
        return JSONResponse({"error": "no input provided"}, status_code=400)
    embs = embedder.encode(texts, batch_size=16)
    return embs

@app.post("/")
async def root(request: Request):
    data = await request.json()
    embs = handle_request_payload(data)
    return embs

@app.post("/v1/embeddings")
async def v1_embeddings(request: Request):
    data = await request.json()
    texts = extract_texts(data)
    if texts is None or len(texts) == 0:
        return JSONResponse({"error": "no input provided"}, status_code=400)
    embs = embedder.encode(texts, batch_size=16)
    if isinstance(data, dict) and "model" in data:
        return build_openai_response(embs)
    return embs