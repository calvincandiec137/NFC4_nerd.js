import os
import json
import numpy as np
import faiss  # type: ignore
import requests # type: ignore
from tqdm import tqdm  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

# Configs
DOCS_JSON_PATH = "./database/sample_json.json"
OUTPUT_DIR = "./embeddings"
EMBED_FILE = os.path.join(OUTPUT_DIR, "vectors.npy")
INDEX_FILE = os.path.join(OUTPUT_DIR, "index.faiss")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters
EMBED_MODEL = SentenceTransformer("BAAI/bge-m3")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_embedding(text: str, model: SentenceTransformer):
    """Generates an embedding for a single text string using the loaded model."""
    if not text.strip():
        print("[⚠️] Warning: Empty text passed to embedding function")
        return None
    try:
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"[❌] Error generating embedding: {e}")
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_index_from_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"[❌] Failed to load JSON file: {e}")
        return

    if not json_data:
        print("[❌] JSON file is empty")
        return

    # Extract documents from the contextual_keypoints array
    documents = json_data.get("contextual_keypoints", [])
    
    if not documents:
        print("[❌] No documents found in contextual_keypoints array")
        return

    print(f"📄 Found {len(documents)} documents to process")
    
    # ✅ REPLACE WITH THIS EFFICIENT BATCH-PROCESSING BLOCK
    all_chunks = []
    metadata = []
    
    for doc in tqdm(documents, desc="🔧 Preparing Chunks"):
        if not isinstance(doc, dict):
            print("[⚠️] Skipping non-dictionary document")
            continue
        
        doc_id = doc.get("section_number", "unknown")
        section_title = doc.get("title", "Untitled Section")
        content = doc.get("keypoints", "")

        if not content.strip():
            continue

        if "theme" in doc:
            content += f"\nTheme: {doc['theme']}"
        if "location" in doc:
            content += f"\nLocation: {doc['location']}"

        chunks = chunk_text(content)
        if not chunks:
            continue

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "doc_id": doc_id,
                "section_title": section_title,
                "location": doc.get("location", ""),
                "theme": doc.get("theme", ""),
                "chunk_index": i,
                "text": chunk,
                "original_chunk_id": doc.get("original_chunk_id", ""),
                "keypoints_length": doc.get("keypoints_length", 0)
            })

    if not all_chunks:
        print("[❌] No valid text chunks to embed.")
        return

    # --- BATCH EMBEDDING ---
    print(f"🚀 Genera~~ng embeddings for {len(all_chunks)} chunks in one batch...")
    vectors = EMBED_MODEL.encode(all_chunks, 
                           batch_size=32,
                           show_progress_bar=True, 
                           normalize_embeddings=True)

    if vectors.size == 0:
        print("[❌] No embeddings were created. Possible reasons:")
        print("- No valid content in documents")
        print("- Network issues")
        return

    print(f"✅ Created {len(vectors)} embeddings from {len(documents)} documents")
    
    vectors = np.array(vectors, dtype=np.float32)
    np.save(EMBED_FILE, vectors)

    dims = vectors.shape[1]
    index = faiss.IndexFlatL2(dims)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved embeddings to {EMBED_FILE}")
    print(f"✅ Saved index to {INDEX_FILE}")
    print(f"✅ Saved metadata to {METADATA_FILE}")

def rag_main():
    print("🚀 Starting RAG index building process")
    print(f"📂 Loading documents from {DOCS_JSON_PATH}")
    build_index_from_json(DOCS_JSON_PATH)


if __name__ == "__main__":
    rag_main()