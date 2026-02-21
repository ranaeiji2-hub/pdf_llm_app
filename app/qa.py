import faiss
import pickle
import requests
import numpy as np

# push 02210933

# ---------- Ollama ----------
OLLAMA_EMBED_URL = "http://host.docker.internal:11434/api/embeddings"
OLLAMA_CHAT_URL  = "http://host.docker.internal:11434/api/chat"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "qwen2.5:3b-instruct"

# ---------- RAG ----------
INDEX_PATH = "index.faiss"
TEXTS_PATH = "texts.pkl"
TOP_K = 3

# ---------- Embedding ----------
def embed(text: str) -> np.ndarray:
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )
    return np.array(r.json()["embedding"], dtype="float32")

# ---------- LLM ----------
def ask_llm(prompt: str) -> str:
    r = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "options": {
                "num_predict": 256,
                "temperature": 0.1,
                "top_k": 2,
                "top_p": 0.8
            }
        },
        timeout=180
    )
    data = r.json()
    return data["message"]["content"]

# ---------- Load ----------
index = faiss.read_index(INDEX_PATH)
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

print("🚀 Fast RAG ready（exitで終了）")

# ---------- Loop ----------
while True:
    query = input("\n> ")
    if query.lower() in ("exit", "quit"):
        break

    # 1️⃣ Embedding
    qvec = embed(query).reshape(1, -1)

    # 2️⃣ FAISSで関連資料取得
    _, ids = index.search(qvec, TOP_K)
    context = "\n\n".join(texts[i] for i in ids[0])

    # 3️⃣ 単一質問用 prompt
    prompt = f"""
以下の資料をもとに質問に答えてください。
資料に無いことは「わかりません」と答えてください。

### 資料
{context}

### 質問
{query}
"""

    # 4️⃣ LLMに問い合わせ
    answer = ask_llm(prompt)

    # 5️⃣ コンソールに出力
    print(answer)
