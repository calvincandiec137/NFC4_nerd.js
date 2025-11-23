import numpy as np
import faiss  # type: ignore
import json
import os
from dotenv import load_dotenv # type: ignore 
from groq import Groq  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from collections import deque
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()
# --- CONFIGURATION ---
# File paths
VEC_PATH = "./embeddings/vectors.npy"
INDEX_PATH = "./embeddings/index.faiss"
META_PATH = "./embeddings/metadata.json"

# Model names
GEN_MODEL = "llama-3.1-8b-instant" # Groq model for generation

# --- 🧠 LOAD MODELS AND CLIENTS ONCE AT STARTUP ---
try:
    print("Loading embedding model (BAAI/bge-m3)...")
    # This loads the model from Hugging Face and caches it locally.
    st_model = SentenceTransformer("BAAI/bge-m3")
    
    print("Configuring Groq client...")
    # This initializes the Groq client using your environment variable.
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    print("✅ Models and clients loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load models/clients. Ensure GROQ_API_KEY is set and libraries are installed. Error: {e}")
    exit()

# --- Load FAISS index and metadata ---
try:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Loaded FAISS index with {index.ntotal} vectors.")
except Exception as e:
    print(f"❌ Failed to load FAISS index or metadata from '{INDEX_PATH}' and '{META_PATH}'. Error: {e}")
    exit()

# --- CONVERSATION CONTEXT ---
context_window = deque(maxlen=5)

def get_embedding(text):
    """Generates an embedding using the pre-loaded SentenceTransformer model."""
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
    # The BGE model with normalize_embeddings=True already prepares for cosine similarity
    
    distances, indices = index.search(query_vec, top_k)
    results = []
    for i, dist in zip(indices[0], distances[0]):
        match = metadata[i]
        # Distance in FAISS IndexFlatL2 is squared L2, but after normalization, it's related to cosine similarity.
        # A lower distance means higher similarity. 1 - (dist / 2) is a way to map it.
        match["similarity"] = (1 - (dist / 2)) * 100
        results.append(match)

    return results

def context_add(message: str):
    """Add a message to the context window."""
    context_window.append(message)

def context_extract() -> str:
    """Extract all messages as a single string context."""
    return "\n".join(context_window)

def generate_answer(query, context):
    """Generate a streamed answer from Groq."""
    response_buffer = []
    # context_history is handled by the context_window
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
        print(f"\n[❌] Error generating response: {e}")
        return None

def clean_keypoints(text):
    # Removes the repetitive boilerplate from your JSON
    lines = text.split('\n')
    # Filter out the headers and empty lines
    clean_lines = [line for line in lines if not line.strip().startswith(('**', 'KEYPOINT', '* ', '\t+')) and line.strip()]
    return '\n'.join(clean_lines)


def res_main(query):
    """Main function to handle a user query from retrieval to generation."""
    answer = ""
    try:
        if not query.strip():
            print("⚠️ Please enter a valid question.")
            return

        top_chunks = retrieve_top_chunks(query, top_k=6)

        if not top_chunks:
            print("[⚠️] No relevant chunks found.\n")
            return

        context = "\n\n---\n\n".join([clean_keypoints(chunk["text"]) for chunk in top_chunks])

       # print(f"\n📄 Top {len(top_chunks)} Sections Retrieved:")
        #for i, chunk in enumerate(top_chunks, start=1):
         #   print(f"[{i}] Section: {chunk.get('doc_id', 'N/A')} | Sim: {chunk['similarity']:.2f}% | Loc: {chunk.get('location', 'N/A')}")

        answer = generate_answer(query, context)

        # Prepare data for JSON output, ensuring numpy types are converted
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

       # print("📝 Saved last query result for PDF highlighting.")
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

# --- Command Line Interaction Loop ---
if __name__ == "__main__":
   main()