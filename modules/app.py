from fastapi import FastAPI
from pydantic import BaseModel
import base64
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadPDF(BaseModel):
    data: str
    name: str

@app.get("/")
def root():
    return {"message": "Rag is Running"}

@app.post("/upload")
def upload(payload: UploadPDF):
    file = base64.b64decode(payload.data)

    if len(file) > 10 * 1024 * 1024:
        return {"status": "file too large"}

    BASE_DIR = "./database"

    # ---------------- FIX: ensure folder exists ----------------
    os.makedirs(BASE_DIR, exist_ok=True)
    # -----------------------------------------------------------

    # Clear folder
    for entry in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    # Save file
    with open(f"{BASE_DIR}/{payload.name}", "wb") as f:
        f.write(file)

    return {"status": "file saved"}


@app.get("/prepare")
def prepare():
    import modules.extract_text as extract_text
    from modules.Remote_Rag import ingest_chunks, sanitize_chunks

    # ---------------- FIX: extract_text must not fail ----------------
    os.makedirs("./database", exist_ok=True)
    os.makedirs("./embeddings", exist_ok=True)
    # ----------------------------------------------------------------

    extract_text.main()
    chunks = sanitize_chunks(extract_text.structured)
    ingest_chunks(chunks)

    return {"status": "ingestion complete"}


@app.get("/ask")
def ask_question(q: str):
    from modules.Groq_res import rag_answer
    answer = rag_answer(q)
    return {"question": q, "answer": answer}
