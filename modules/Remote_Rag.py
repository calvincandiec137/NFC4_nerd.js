import os
import time
from dotenv import load_dotenv
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

load_dotenv()

vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

collection_name = "DoChat"

def create_collection(dim):
    names = [c.name for c in qdrant.get_collections().collections]
    if collection_name not in names:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

def embed_batch(texts):
    BATCH = 800
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        print(f"Embedding batch {i//BATCH + 1} with {len(batch)} items...")
        r = vo.embed(batch, model="voyage-3.5")
        all_vecs.extend(r.embeddings)
        time.sleep(3)
    return all_vecs

def upsert_in_batches(points):
    BATCH = 10
    for i in range(0, len(points), BATCH):
        batch = points[i:i + BATCH]
        print(f"Upserting Qdrant batch {i//BATCH + 1} with {len(batch)} points...")
        qdrant.upsert(collection_name=collection_name, points=batch)
        time.sleep(1)

def ingest_chunks(structured):
    texts = [c["text"] for c in structured]
    vectors = embed_batch(texts)
    dim = len(vectors[0])
    create_collection(dim)

    pts = []
    for i, (c, v) in enumerate(zip(structured, vectors)):
        pts.append(
            PointStruct(
                id=i,
                vector=v,
                payload={
                    "text": c["text"],
                    "source": c["source"],
                    "chunk_index": c["chunk_index"]
                }
            )
        )
    upsert_in_batches(pts)

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
    import extract_text
    extract_text.main()
    chunks = sanitize_chunks(extract_text.structured)
    ingest_chunks(chunks)
