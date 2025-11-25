# modules/extract_text.py
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

PATH = "./database"
structured = []

def main():
    """
    - ensure PATH exists
    - iterate PDF files and extract text
    - split text into small chunks using RecursiveCharacterTextSplitter
    """
    os.makedirs(PATH, exist_ok=True)

    global structured
    structured = []

    for fname in os.listdir(PATH):
        if not fname.lower().endswith(".pdf"):
            # skip non-pdf: keep behavior consistent with your previous logic
            continue

        full_path = os.path.join(PATH, fname)
        try:
            doc = fitz.open(full_path)
        except Exception:
            continue

        text = ""
        for page in doc:
            try:
                text += page.get_text()
            except Exception:
                # skip problematic pages
                continue

        # splitter: small chunk size to ensure safe embedding on small CPU instances
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_text(text)

        for i, chunk in enumerate(texts):
            structured.append({
                "id": f"{fname}__{i}",
                "text": chunk,
                "source": fname,
                "chunk_index": i
            })

if __name__ == "__main__":
    main()
