import os
from dotenv import load_dotenv
import voyageai
from groq import Groq
from qdrant_client import QdrantClient

load_dotenv()

vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

collection = "DoChat"

def embed_query(text):
    r = vo.embed([text], model="voyage-3.5")
    return r.embeddings[0]

def search_qdrant(query_embedding, top_k=5):
    res = qdrant.query_points(
        collection_name=collection,
        query=query_embedding,
        limit=top_k
    )
    return res.points


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
    q_vec = embed_query(question)
    hits = search_qdrant(q_vec, top_k=5)

    context = "\n\n".join([h.payload["text"] for h in hits])
    answer = generate_answer(question, context)
    return answer

if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        print("\nAnswer:\n", rag_answer(q))
