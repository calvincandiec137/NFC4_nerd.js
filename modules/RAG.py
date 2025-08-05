import os
import json
import numpy as np
import faiss  # type: ignore
import requests
from tqdm import tqdm  # type: ignore

# Configs
DOCS_JSON_PATH = "./database/sample_document_enhanced_analysis_20250806_001238.json"
OUTPUT_DIR = "./embeddings"
EMBED_FILE = os.path.join(OUTPUT_DIR, "vectors.npy")
INDEX_FILE = os.path.join(OUTPUT_DIR, "index.faiss")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters
EMBED_MODEL = "nomic-embed-text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_embedding(text: str):
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    try:
        response = requests.post("http://localhost:11434/api/embeddings", json=payload)
        response.raise_for_status()
        embedding = response.json()["embedding"]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"[❌] Error generating embedding: {e}")
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_index_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    documents = json_data.get("contextual_summaries", [])
    vectors = []
    metadata = []

    for doc in tqdm(documents, desc="🔧 Processing Documents"):
        doc_id = doc.get("section_number", "unknown")
        section_title = doc.get("title", "Untitled Section")
        content = doc.get("summary", "")

        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            vec = fetch_embedding(chunk)
            if vec is not None:
                vectors.append(vec)
                metadata.append({
                    "doc_id": doc_id,
                    "section_title": section_title,
                    "chunk_index": i,
                    "text": chunk
                })

    if not vectors:
        print("[⚠️] No embeddings were created. Check input data.")
        return

    vectors = np.array(vectors, dtype=np.float32)
    np.save(EMBED_FILE, vectors)

    dims = vectors.shape[1]
    index = faiss.IndexFlatL2(dims)
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Embedding, index, and metadata saved.")

if __name__ == "__main__":
    build_index_from_json(DOCS_JSON_PATH)
