import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize Qdrant client
# Supports both local (in-memory) and Qdrant Cloud
qdrant_url = os.environ.get("QDRANT_URL", ":memory:")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")

if qdrant_url == ":memory:":
    qdrant_client = QdrantClient(":memory:")
else:
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

collection_name = "dochat"

# Create collection if it doesn't exist
try:
    qdrant_client.get_collection(collection_name)
except:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

def embed_batch(texts):
    """Embed texts using Google Gemini text-embedding-004 model"""
    BATCH = 100  # Gemini can handle larger batches
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        print(f"Embedding batch {i//BATCH + 1} with {len(batch)} items...")
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="retrieval_document"
            )
            all_vecs.extend(result['embedding'])
            time.sleep(0.5)  # Small delay to respect rate limits
        except Exception as e:
            print(f"Error embedding batch: {e}")
            # Retry with smaller batch if failed
            for text in batch:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                all_vecs.append(result['embedding'])
                time.sleep(0.2)
    return all_vecs

def upsert_qdrant(points):
    """Upload points to Qdrant"""
    BATCH = 100
    for i in range(0, len(points), BATCH):
        batch = points[i:i + BATCH]
        print(f"Upserting Qdrant batch {i//BATCH + 1} with {len(batch)} points...")
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
        time.sleep(0.5)

def ingest_chunks(structured):
    """Ingest document chunks into Qdrant"""
    texts = [c["text"] for c in structured]
    vectors = embed_batch(texts)

    points = []
    for i, (c, v) in enumerate(zip(structured, vectors)):
        points.append(
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
    upsert_qdrant(points)

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
