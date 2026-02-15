import os
import json
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


INDEX_PATH = "./embeddings/index.faiss"
META_PATH = "./embeddings/metadata.json"


EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GROQ_MODEL = "llama-3.1-8b-instant" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
reranker = CrossEncoder(RERANK_MODEL)

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def embed_query(query: str):
    return embed_model.encode(
        query,
        normalize_embeddings=True
    ).astype(np.float32).reshape(1, -1)

def retrieve(query: str, top_k: int = 20):
    qv = embed_query(query)
    scores, ids = index.search(qv, top_k)
    return [metadata[i] for i in ids[0]]

def rerank(query: str, docs, top_n: int = 5):
    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


def generate_answer(query: str, context: str) -> str:
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{query}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()


def ask(query: str):
    retrieved = retrieve(query)
    reranked = rerank(query, retrieved)
    context = "\n\n---\n\n".join(d["text"] for d in reranked)

    answer = generate_answer(query, context)
    print("\nAnswer:\n", answer)


def interactive():
    print("RAG system ready (Groq backend).")
    print("Type a question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            q = input("\n> ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            ask(q)
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit.")
            continue

def main():
    import sys
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        interactive()

if __name__ == "__main__":
    main()
