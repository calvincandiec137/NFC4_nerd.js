import os
import time
import requests
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

JINA_API_KEY = os.environ["JINA_API_KEY"]

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "dochat"

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Jina v3 is also 1536
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

JINA_URL = "https://api.jina.ai/v1/embeddings"

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

    data = r.json()["data"]
    return [d["embedding"] for d in data]


def embed_batch(texts):
    BATCH = 800
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        print(f"Embedding batch {i//BATCH + 1} with {len(batch)} items...")
        vecs = jina_embed(batch, task="retrieval.passage")
        all_vecs.extend(vecs)
        time.sleep(1)
    return all_vecs


def upsert_pinecone(points):
    BATCH = 100
    for i in range(0, len(points), BATCH):
        batch = points[i:i + BATCH]
        print(f"Upserting Pinecone batch {i//BATCH + 1} with {len(batch)} points...")
        index.upsert(vectors=batch)
        time.sleep(0.5)


def ingest_chunks(structured):
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


def sanitize_chunks(structured):
    clean = []
    seen = set()
    for c in structured:
        t = c.get("text")
        if not isinstance(t, str):
            continue
        t2 = t.strip()
        if len(t2) < 20:
            continue
        if t2 in seen:
            continue
        seen.add(t2)
        clean.append({**c, "text": t2})
    return clean


if __name__ == "__main__":
    import modules.extract_text as extract_text
    extract_text.main()
    chunks = sanitize_chunks(extract_text.structured)
    ingest_chunks(chunks)
    print("Ingestion complete.")
    print(f"Total chunks ingested: {len(chunks)}")
    