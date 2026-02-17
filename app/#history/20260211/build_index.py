import pdfplumber
import faiss
import requests
import numpy as np
import pickle

OLLAMA_URL = "http://host.docker.internal:11434/api/embeddings"
MODEL = "nomic-embed-text"
PDF_PATH = "sample.pdf"
INDEX_PATH = "index.faiss"
TEXTS_PATH = "texts.pkl"
MAX_CHARS = 300  # 重要：埋め込みの上限対策


def embed(text: str) -> np.ndarray:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        },
        timeout=120
    )
    data = r.json()
    if "embedding" not in data:
        raise RuntimeError(f"Ollama error: {data}")
    return np.array(data["embedding"], dtype="float32")


def split_text(text: str, max_chars: int = MAX_CHARS):
    return [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
        if text[i:i + max_chars].strip()
    ]


# ---------- PDF 読み込み ----------
chunks = []

with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            chunks.extend(split_text(text))

if not chunks:
    raise RuntimeError("PDFからテキストを取得できませんでした")


# ---------- Embedding ----------
vectors = []
valid_texts = []

for t in chunks:
    try:
        v = embed(t)
        vectors.append(v)
        valid_texts.append(t)
    except Exception as e:
        print("skip chunk:", str(e))

if not vectors:
    raise RuntimeError("埋め込みベクトルが1件も作成できませんでした")

vectors = np.vstack(vectors).astype("float32")


# ---------- FAISS ----------
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, INDEX_PATH)
with open(TEXTS_PATH, "wb") as f:
    pickle.dump(valid_texts, f)

print("index built")
print(f"chunks: {len(valid_texts)}")
