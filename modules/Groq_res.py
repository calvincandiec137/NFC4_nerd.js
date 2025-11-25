import os
import requests
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

JINA_API_KEY = os.environ["JINA_API_KEY"]

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index = pc.Index("dochat")

JINA_URL = "https://api.jina.ai/v1/embeddings"


def jina_embed_query(text):
    payload = {
        "model": "jina-embeddings-v3",
        "task": "retrieval.query",
        "input": [text]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }

    r = requests.post(JINA_URL, json=payload, headers=headers)
    r.raise_for_status()

    return r.json()["data"][0]["embedding"]


def search_pinecone(query_vec, top_k=5):
    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True
    )
    return res.matches


def generate_answer(query, context):
    prompt = f"""Use ONLY the context to answer.

Context:
{context}

Question: {query}

Answer:"""

    r = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return r.choices[0].message.content.strip()


def rag_answer(question):
    q_vec = jina_embed_query(question)
    hits = search_pinecone(q_vec, top_k=5)

    context = "\n\n".join([h.metadata["text"] for h in hits])
    answer = generate_answer(question, context)
    return answer


if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        print("\nAnswer:\n", rag_answer(q))
