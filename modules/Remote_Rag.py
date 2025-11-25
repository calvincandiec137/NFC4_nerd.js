import os
import time
import requests
from pinecone import Pinecone, ServerlessSpec

# -------------------------------
# ENV
# -------------------------------
JINA_URL = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = os.environ["JINA_API_KEY"]

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "dochat"

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


# ---------------------------------------------------------
# MINIMAL FIX 1 → CLEAN TEXT (nothing else changed)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# MINIMAL FIX 2 → CUT LONG TEXTS FOR JINA
# (keeps your ingest logic EXACT)
# ---------------------------------------------------------
def cut_for_jina(t: str, max_len=1500):
    t = clean_text(t)
    if not t:
        return []

    if len(t) <= max_len:
        return [t]

    out = []
    while len(t) > max_len:
        cut = t.rfind(" ", 0, max_len)
        if cut == -1:
            cut = max_len
        out.append(t[:cut].strip())
        t = t[cut:].strip()

    if t:
        out.append(t)

    return out


# ---------------------------------------------------------
# MINIMAL FIX 3 → replace sanitize output with cut chunks
# ---------------------------------------------------------
def sanitize_chunks(structured):
    clean = []
    seen = set()

    for c in structured:
        t = c.get("text")
        t = clean_text(t)
        if not t:
            continue

        if t in seen:
            continue
        seen.add(t)

        # CUT HERE (NEW FIX)
        parts = cut_for_jina(t)

        for p in parts:
            clean.append({
                **c,
                "text": p          # KEEP YOUR LOGIC, JUST REPLACE TEXT W/ CUT
            })

    return clean


# ---------------------------------------------------------
# JINA call (unchanged)
# ---------------------------------------------------------
def jina_embed(texts, task="retrieval.passage"):
    payload = {
        "model": "jina-embeddings-v3",
        "task": task,
        "input": texts
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }

    r = requests.post(JINA_URL, json=payload, headers=headers)
    r.raise_for_status()

    return [d["embedding"] for d in r.json()["data"]]


# ---------------------------------------------------------
# YOUR ORIGINAL embed_batch (only debug added)
# ---------------------------------------------------------
def embed_batch(texts):
    BATCH = 800    # YOUR original value
    all_vecs = []

    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        print(f"Embedding batch {i//BATCH + 1} with {len(batch)} items...")

        try:
            vecs = jina_embed(batch, task="retrieval.passage")
        except Exception as e:
            print("JINA ERROR ON BATCH:", i)
            print(batch)
            raise e

        all_vecs.extend(vecs)
        time.sleep(2)     # YOUR original sleep

    return all_vecs


# ---------------------------------------------------------
# YOUR ORIGINAL Pinecone upsert
# ---------------------------------------------------------
def upsert_pinecone(points):
    BATCH = 100
    for i in range(0, len(points), BATCH):
        batch = points[i:i + BATCH]
        print(f"Upserting Pinecone batch {i//BATCH + 1} with {len(batch)} points...")
        index.upsert(vectors=batch)
        time.sleep(1)


# ---------------------------------------------------------
# YOUR ORIGINAL ingest function
# ---------------------------------------------------------
def ingest_chunks(structured):
    structured = sanitize_chunks(structured)  # ← ONLY CHANGE
    texts = [c["text"] for c in structured]

    vectors = embed_batch(texts)

    payloads = []
    for i, (c, v) in enumerate(zip(structured, vectors)):
        payloads.append({
            "id": str(i),
            "values": v,
            "metadata": {
                "text": c["text"],
                "source": c["source"],
                "chunk_index": c["chunk_index"]
            }
        })

    upsert_pinecone(payloads)
