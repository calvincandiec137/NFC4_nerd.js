import numpy as np
import faiss  # type: ignore
import json
import os
import requests

# File paths
VEC_PATH = "./embeddings/vectors.npy"
INDEX_PATH = "./embeddings/index.faiss"
META_PATH = "./embeddings/metadata.json"

# Ollama model names
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3"

# Load FAISS index and metadata
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Get embedding from Ollama (nomic-embed-text)
def get_embedding(text):
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
        print(f"[❌] Embedding Error: {e}")
        return None

# Retrieve top-k relevant chunks from FAISS
def retrieve_top_chunks(query, top_k=3):
    query_vec = get_embedding(query)
    if query_vec is None:
        print("[⚠️] Could not get embedding for the query.")
        return []

    # Ensure shape is (1, dim)
    query_vec = query_vec.reshape(1, -1)

    distances, indices = index.search(query_vec, top_k)
    results = []
    for i, dist in zip(indices[0], distances[0]):
        match = metadata[i]
        match["similarity"] = 100 - dist  # Lower distance = higher similarity
        results.append(match)

    return results

# Generate streamed answer from LLaMA 3 using Ollama
def generate_answer(query, context):
    prompt = f"""You are an intelligent assistant. Use the given context to answer the user query in 2 lines max.

Context:
{context}

Query:
{query}

Only answer based on the context. If it is not answerable, say you don't know.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": GEN_MODEL,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )
        print("\n💬 Answer:", end=" ", flush=True)
        for chunk in response.iter_lines():
            if chunk:
                content = json.loads(chunk.decode("utf-8"))["response"]
                print(content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\n[❌] Error generating response: {e}")

# CLI Interaction Loop
if __name__ == "__main__":
    print("\n🤖 AI Document Assistant (powered by Ollama)\n")
    while True:
        try:
            query = input("🔎 Ask something:\n> ").strip()
            if not query:
                print("⚠️ Please enter a valid question.")
                continue

            top_chunks = retrieve_top_chunks(query, top_k=3)
            if not top_chunks:
                print("[⚠️] No relevant chunks found.\n")
                continue

            context = "\n\n---\n\n".join([chunk["text"] for chunk in top_chunks])

            print(f"\n📄 Top {len(top_chunks)} Sections Retrieved:")
            for i, chunk in enumerate(top_chunks, start=1):
                print(f"[{i}] Section: {chunk['doc_id']} | Similarity: {chunk['similarity']:.2f}%")

            generate_answer(query, context)

        except KeyboardInterrupt:
            print("\n👋 Exiting assistant.")
            break