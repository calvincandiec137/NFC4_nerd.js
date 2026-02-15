import os
import json
import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DOCS_JSON_PATH = "./database/sample_json.json"
OUTPUT_DIR = "./embeddings"
INDEX_FILE = os.path.join(OUTPUT_DIR, "index.faiss")
META_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
BATCH_SIZE = 64

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def main():
    with open(DOCS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("contextual_keypoints", [])
    texts, metadata = [], []

    for doc in tqdm(docs, desc="Chunking"):
        content = doc.get("keypoints", "").strip()
        if not content:
            continue

        if doc.get("theme"):
            content += f"\nTheme: {doc['theme']}"
        if doc.get("location"):
            content += f"\nLocation: {doc['location']}"

        for i, chunk in enumerate(chunk_text(content)):
            texts.append(chunk)
            metadata.append({
                "doc_id": doc.get("section_number"),
                "title": doc.get("title"),
                "chunk_id": i,
                "text": chunk
            })

    embeddings = embed_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Device used for embeddings: {DEVICE}")
    print("FAISS CPU index created")
    print(f"Indexed {index.ntotal} chunks")

if __name__ == "__main__":
    main()
