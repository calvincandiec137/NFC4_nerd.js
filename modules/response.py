import numpy as np
import faiss #type:ignore
import json
import os
from sentence_transformers import SentenceTransformer #type:ignore
import google.generativeai as genai #type:ignore
from dotenv import load_dotenv

load_dotenv()


VEC_PATH = "./embeddings/vectors.npy"
INDEX_PATH = "./embeddings/index.faiss"
META_PATH = "./embeddings/metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = genai.GenerativeModel(GEMINI_MODEL)

embedding_model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def get_embedding(text):
    vec = embedding_model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)

def retrieve_top_chunk(query, top_k=1):
    query_vec = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        match = metadata[i]
        match["similarity"] = 100 - dist
        results.append(match)

    return results

def generate_answer(query, context):
    prompt = f"""You are an intelligent assistant. Use the given context to answer the user query in 2 lines max.

Context:
{context}

Query:
{query}

Only answer based on the context. If it is not answerable, say you don't know.
"""
    response = llm.generate_content(prompt)
    return response.text.strip()

if __name__ == "__main__":
    while True:
        query = input("Ask something:\n")
        top = retrieve_top_chunk(query, top_k=1)[0]
        print(f"\nRetrieved Chunk (Similarity: {top['similarity']:.2f}%)\nSection: {top['doc_id']}\n")
        answer = generate_answer(query, top["text"])
        print(f"Answer: {answer}\n")
