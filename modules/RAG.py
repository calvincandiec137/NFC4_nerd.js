import os
import json
import numpy as np
import faiss #type:ignore
from tqdm import tqdm #type:ignore
from sentence_transformers import SentenceTransformer #type:ignore

# Configs
MODEL_NAME = "all-MiniLM-L6-v2"
DOCS_JSON_PATH = "./database/summaries.json"
OUTPUT_DIR = "./embeddings"
EMBED_FILE = os.path.join(OUTPUT_DIR, "vectors.npy")
INDEX_FILE = os.path.join(OUTPUT_DIR, "index.faiss")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

def fetch_embedding(text):
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)

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
        documents = json.load(f)

    vectors = []
    metadata = []

    for doc in tqdm(documents, desc="Processing Documents"):
        doc_id = doc.get("section_title", "unknown")
        content = doc.get("summary", "")
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            vec = fetch_embedding(chunk)
            vectors.append(vec)
            metadata.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "text": chunk
            })

    if not vectors:
        print("No embeddings were created. Check your input data.")
        return

    vectors = np.array(vectors, dtype=np.float32)
    np.save(EMBED_FILE, vectors)

    dims = vectors.shape[1]
    index = faiss.IndexFlatL2(dims)
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Embedding, index, and metadata saved.")

if __name__ == "__main__":
    build_index_from_json(DOCS_JSON_PATH)
