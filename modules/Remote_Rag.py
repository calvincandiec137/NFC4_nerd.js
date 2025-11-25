import os
import time
from dotenv import load_dotenv
import voyageai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

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

def embed_batch(texts):
    BATCH = 20
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        print(f"Embedding batch {i//BATCH + 1} with {len(batch)} items...")
        r = vo.embed(batch, model="voyage-3.5")
        all_vecs.extend(r.embeddings)
        time.sleep(2)
    return all_vecs

def upsert_pinecone(points):
    BATCH = 100
    for i in range(0, len(points), BATCH):
        batch = points[i:i + BATCH]
        print(f"Upserting Pinecone batch {i//BATCH + 1} with {len(batch)} points...")
        index.upsert(vectors=batch)
        time.sleep(1)

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
