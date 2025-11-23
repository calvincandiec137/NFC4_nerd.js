import numpy as np
import faiss
import json
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from collections import deque
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

VEC_PATH = "./embeddings/vectors.npy"
INDEX_PATH = "./embeddings/index.faiss"
META_PATH = "./embeddings/metadata.json"

GEN_MODEL = "llama-3.1-8b-instant"

try:
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Configuring Groq client...")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Models and clients loaded.")
except Exception as e:
    print(f"Failed to load models/clients. Error: {e}")
    exit()

try:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded FAISS index with {index.ntotal} vectors.")
except Exception as e:
    print(f"Failed to load FAISS index or metadata. Error: {e}")
    exit()

context_window = deque(maxlen=5)

def get_embedding(text):
    """Return embedding for `text`."""
    try:
        embedding = st_model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"[❌] Embedding Error: {e}")
        return None

def retrieve_top_chunks(query, top_k=6):
    """Retrieve top-k relevant chunks from FAISS based on query similarity."""
    query_vec = get_embedding(query)
    if query_vec is None:
        print("[⚠️] Could not get embedding for the query.")
        return []

    query_vec = query_vec.reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for i, dist in zip(indices[0], distances[0]):
        match = metadata[i]
        match["similarity"] = (1 - (dist / 2)) * 100
        results.append(match)

    return results

def context_add(message: str):
    """Append a message to the context window."""
    context_window.append(message)

def context_extract() -> str:
    """Return joined context as a single string."""
    return "\n".join(context_window)

def generate_answer(query, context):
    """Stream an answer from Groq."""
    response_buffer = []
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's query.
        Your answer should be a concise summary of the information found in the context.

        If the context does not contain the answer, state that the information is not available.

        Context:
        {context}

        User Query:
        {query}
        """

    try:
        print("\n💬 Answer:", end=" ", flush=True)
        stream = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GEN_MODEL,
            stream=True,
            temperature=0.2,
            max_tokens=150
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                response_buffer.append(content)
        
        print("\n")
        full_response = "".join(response_buffer)
        context_add(f"Assistant: {full_response}")
        return full_response
    except Exception as e:
        print(f"\nFailed to generate response: {e}")
        return None

def clean_keypoints(text):
    lines = text.split('\n')
    clean_lines = [line for line in lines if not line.strip().startswith(('**', 'KEYPOINT', '* ', '\t+')) and line.strip()]
    return '\n'.join(clean_lines)


def res_main(query):
    """Retrieve chunks for `query` and generate an answer."""
    answer = ""
    try:
        if not query.strip():
            print("Enter a question.")
            return

        top_chunks = retrieve_top_chunks(query, top_k=6)

        if not top_chunks:
            print("[⚠️] No relevant chunks found.\n")
            return

        context = "\n\n---\n\n".join([clean_keypoints(chunk["text"]) for chunk in top_chunks])

        answer = generate_answer(query, context)

        for chunk in top_chunks:
            if isinstance(chunk.get("similarity"), (np.floating, float)):
                chunk["similarity"] = float(chunk["similarity"])

        session_data = {
            "query": query,
            "rag_response": answer,
            "chunks": top_chunks
        }

        with open("./modules/last_query_result.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        return answer

    except KeyboardInterrupt:
        print("\n👋 Exiting assistant.")
        return None
def main():
    while True:
        try:
            user_query = input("\n❓ Ask a question (or type 'exit' to quit):\n> ").strip()
            if user_query.lower() in ("exit", "quit"):
                print("👋 Goodbye.")
                break
            res_main(user_query)
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user.")
            break

if __name__ == "__main__":
   main()