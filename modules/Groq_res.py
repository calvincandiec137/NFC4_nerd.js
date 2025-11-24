import os
from dotenv import load_dotenv
import voyageai
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "dochat"
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

def embed_query(text):
    r = vo.embed([text], model="voyage-3.5")
    return r.embeddings[0]

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
    q_vec = embed_query(question)
    hits = search_pinecone(q_vec, top_k=5)

    context = "\n\n".join([h.metadata["text"] for h in hits])
    answer = generate_answer(question, context)
    return answer

if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        print("\nAnswer:\n", rag_answer(q))
