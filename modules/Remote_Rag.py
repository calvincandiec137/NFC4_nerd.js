# modules/Remote_Rag.py
import os
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# environment / config
MODEL_NAME = "all-MiniLM-L6-v2"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = "dochat"
PINECONE_DIM = 384

# init model once at import
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

# init pinecone client (keeps your previous Pinecone usage style)
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=PINECONE_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX)

# ---------------------------
# Clean/sanitize text (keeps your original sanitize but stricter)
# ---------------------------
def clean_text(t: str):
    if not isinstance(t, str):
        return None
    t = t.strip()
    if not t:
        return None
    t = t.replace("\x00", "")
    t = t.encode("utf-8", "ignore").decode("utf-8", "ignore")
    t = "".join(ch for ch in t if ch.isprintable() or ch.isspace())
    if len(t) < 20:
        return None
    return t

def sanitize_chunks(structured):
    """
    Keep your original semantics but ensure strings are cleaned and deduped.
    """
    clean = []
    seen = set()
    for c in structured:
        t = c.get("text")
        if not isinstance(t, str):
            continue
        t2 = clean_text(t)
        if not t2:
            continue
        if t2 in seen:
            continue
        seen.add(t2)
        clean.append({**c, "text": t2})
    return clean

# ---------------------------
# Embedding (all-MiniLM)
# ---------------------------
def embed_batch(texts):
    """
    Use local SentenceTransformer to embed texts.
    BATCH tuned for CPU Render; adjust if you have GPU or more RAM.
    """
    model = get_model()
    BATCH = 64   # safe default for CPU — adjust if you have more RAM/CPU
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        print(f"Embedding batch {i//BATCH + 1} with {len(batch)} items...")
        # model.encode returns numpy arrays; keep them as lists for pinecone
        vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        for v in vecs:
            all_vecs.append(v.tolist())
        # be polite on CPU; pause briefly
        time.sleep(0.2)
    return all_vecs

# ---------------------------
# Upsert into Pinecone
# ---------------------------
def upsert_pinecone(points):
    BATCH = 100
    for i in range(0, len(points), BATCH):
        batch = points[i:i + BATCH]
        print(f"Upserting Pinecone batch {i//BATCH + 1} with {len(batch)} points...")
        index.upsert(vectors=batch)
        time.sleep(0.5)

# ---------------------------
# Main ingest function (keeps your original structure)
# ---------------------------
def ingest_chunks(structured):
    """
    structured: list of dicts with keys text, source, chunk_index
    """
    texts = [c["text"] for c in structured]
    vectors = embed_batch(texts)

    payloads = []
    for i, (c, v) in enumerate(zip(structured, vectors)):
        payloads.append({
            "id": str(i),
            "values": v,
            "metadata": {
                "text": c["text"],
                "source": c.get("source", ""),
                "chunk_index": c.get("chunk_index", 0)
            }
        })
    upsert_pinecone(payloads)
