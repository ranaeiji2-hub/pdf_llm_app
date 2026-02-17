import faiss
import pickle
import requests
import numpy as np

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
            "stream": False,   # ★ これを必ず入れる
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

    # 2️⃣ FAISS
    _, ids = index.search(qvec, TOP_K)
    context = "\n\n".join(texts[i] for i in ids[0])

    # 3️⃣ Prompt
    prompt = f"""
以下の資料をもとに、CSV形式で情報を整理してください。
出力ルールは厳格に守ること。

【出力ルール】
- 1行目は必ずヘッダ
- カンマ区切り
- ダブルクォートで囲む
- 情報が無い項目は「不明」
- 説明文や補足は禁止（CSVのみ出力）

【CSVヘッダ】
"車名","車種","全長","全幅","全高","室内長","室内幅","室内高","ホイールベース"

### 資料
{context}

### 質問
{query}
"""

    # 4️⃣ LLM
    answer = ask_llm(prompt)

    # 5️⃣ CSV出力
    OUTPUT_PATH = "/workspaces/pdf_llm_app/output.csv"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(answer)

    print(f"output.csv に出力しました → {OUTPUT_PATH}")
