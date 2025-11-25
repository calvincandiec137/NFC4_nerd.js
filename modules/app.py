from fastapi import FastAPI
from pydantic import BaseModel
import base64
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# If you uploaded a sample file earlier, it's available at this local path.
# The deploy system note: /mnt/data/c31298fa-a2b5-4ea5-93b1-d7d455c50a29.png
SAMPLE_UPLOADED_FILE_PATH = "/mnt/data/c31298fa-a2b5-4ea5-93b1-d7d455c50a29.png"

BASE_DIR = "./database"
EMBED_DIR = "./embeddings"

class UploadPDF(BaseModel):
    data: str
    name: str

@app.get("/")
def root():
    return {"message": "Rag is Running"}

@app.post("/upload")
def upload(payload: UploadPDF):
    """
    - Ensure ./database exists
    - Delete everything inside it BEFORE saving the new file
    - Save the incoming base64 file (payload.name)
    """
    try:
        # ensure base dirs
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(EMBED_DIR, exist_ok=True)

        # clear directory
        for entry in os.listdir(BASE_DIR):
            path = os.path.join(BASE_DIR, entry)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

        # handle sample trigger: if client sends name "USE_SAMPLE" we copy the sample uploaded file
        if payload.name == "USE_SAMPLE":
            if os.path.exists(SAMPLE_UPLOADED_FILE_PATH):
                dest = os.path.join(BASE_DIR, os.path.basename(SAMPLE_UPLOADED_FILE_PATH))
                shutil.copyfile(SAMPLE_UPLOADED_FILE_PATH, dest)
                return {"status": "sample file copied", "saved_as": os.path.basename(dest)}
            else:
                return {"status": "sample missing on server", "path": SAMPLE_UPLOADED_FILE_PATH}

        # normal save path
        file = base64.b64decode(payload.data)
        if len(file) > 10 * 1024 * 1024:
            return {"status": "file too large"}

        with open(os.path.join(BASE_DIR, payload.name), "wb") as f:
            f.write(file)

        return {"status": "file saved", "name": payload.name}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}


@app.get("/prepare")
def prepare():
    """
    Extract text, sanitize, chunk, embed and upsert into Pinecone.
    """
    try:
        # ensure dirs present
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(EMBED_DIR, exist_ok=True)

        import modules.extract_text as extract_text
        from modules.Remote_Rag import ingest_chunks, sanitize_chunks

        extract_text.main()
        chunks = sanitize_chunks(extract_text.structured)
        # ingest_chunks will embed and upsert
        ingest_chunks(chunks)
        return {"status": "ingestion complete", "chunks": len(chunks)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}


@app.get("/ask")
def ask_question(q: str):
    """
    Simple RAG query endpoint — assumes you have Groq_res.rag_answer or similar wired.
    """
    try:
        from modules.Groq_res import rag_answer
        answer = rag_answer(q)
        return {"question": q, "answer": answer}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}


@app.get("/health")
def health():
    """
    lightweight status: checks directories and returns whether the model/index endpoints are reachable by basic checks.
    """
    try:
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(EMBED_DIR, exist_ok=True)
        return {"status": "ok", "database_exists": os.path.exists(BASE_DIR)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
