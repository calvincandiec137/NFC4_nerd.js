import os
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

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

def embed_query(text):
    """Embed query using Google Gemini"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

def search_qdrant(query_vec, top_k=5):
    """Search Qdrant for similar vectors"""
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k
    )
    return results

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
    """Generate RAG answer using Qdrant search and Groq LLM"""
    q_vec = embed_query(question)
    hits = search_qdrant(q_vec, top_k=5)

    context = "\n\n".join([h.payload["text"] for h in hits])
    answer = generate_answer(question, context)
    return answer

if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        print("\nAnswer:\n", rag_answer(q))
