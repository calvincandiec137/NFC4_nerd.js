from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_DIR = "./database"
os.makedirs(DATABASE_DIR, exist_ok=True)

class UploadPDF(BaseModel):
    data: str
    name: str

@app.get("/")
def root():
    return {"message": "DoChat RAG API is Running"}

@app.post("/upload")
def upload(payload: UploadPDF):
    """Upload a PDF file to the database directory"""
    try:
        file_data = base64.b64decode(payload.data)

        # Check file size (10MB limit)
        if len(file_data) > 10 * 1024 * 1024:
            return {"status": "error", "message": "File too large (max 10MB)"}

        # Save file to database directory
        file_path = os.path.join(DATABASE_DIR, payload.name)
        with open(file_path, "wb") as f:
            f.write(file_data)

        return {"status": "success", "message": f"File {payload.name} uploaded successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/prepare")
def prepare():
    """Process the uploaded PDFs and create embeddings"""
    try:
        # Import and run the RAG indexing
        from modules.RAG import main as rag_main
        rag_main()
        
        return {"status": "success", "message": "Documents processed and indexed successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to prepare documents: {str(e)}"}

@app.get("/ask")
def ask_question(q: str):
    """Ask a question and get an answer using RAG"""
    try:
        # Import the response function
        from modules.response_groq import rag_query
        
        answer = rag_query(q)
        return {"question": q, "answer": answer}
    except Exception as e:
        return {"question": q, "answer": f"Error: {str(e)}"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "backend": "online"}
