from fastapi import FastAPI
import faiss
import pickle
import requests
import numpy as np

app = FastAPI()

index = faiss.read_index("index.faiss")
texts = pickle.load(open("texts.pkl", "rb"))

def embed(text):
    r = requests.post(
        "http://host.docker.internal:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return np.array(r.json()["embedding"], dtype="float32")

def llama(prompt):
    r = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    return r.json()["response"]

@app.get("/ask")
def ask(q: str):
    qv = embed(q).reshape(1, -1)
    _, ids = index.search(qv, 3)

    context = "\n".join(texts[i] for i in ids[0])
    prompt = f"{context}\n\n質問: {q}"

    return {"answer": llama(prompt)}
