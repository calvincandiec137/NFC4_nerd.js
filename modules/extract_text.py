import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

PATH = "/home/faizmk/NFC4_nerd.js/database"

structured = []

def main():
    for fname in os.listdir(PATH):
        if not fname.endswith(".pdf"):
            continue

        full_path = os.path.join(PATH, fname)
        doc = pymupdf.open(full_path)

        text = ""
        for page in doc:
            text += page.get_text()

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = splitter.split_text(text)

        for i, chunk in enumerate(texts):
            structured.append({
                "id": i,
                "text": chunk,
                "source": fname,
                "chunk_index": i
            })

if __name__ == "__main__":
    main()
