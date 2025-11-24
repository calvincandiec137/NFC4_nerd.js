from fastapi import FastAPI
from pydantic import BaseModel
import base64

app = FastAPI()

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

    BASE_DIR = "/home/faizmk/NFC4_nerd.js/database"

    with open(f"{BASE_DIR}/{payload.name}", "wb") as f:
        f.write(file)

    return {"status": "file saved"}

@app.get("/prepare")
def prepare():
    import extract_text
    from Remote_Rag import ingest_chunks, sanitize_chunks

    extract_text.main()
    chunks = sanitize_chunks(extract_text.structured)
    ingest_chunks(chunks)
    return {"status": "ingestion complete"}

@app.get("/ask")
def ask_question(q: str):
    from Groq_res import rag_answer
    answer = rag_answer(q)
    return {"question": q, "answer": answer}