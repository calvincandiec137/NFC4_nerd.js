import os
import re
import json
import time
import fitz
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

OUTPUT_JSON = "./database/sample_json.json"

MAX_LLM_CHARS = 12000
MAX_WORKERS = 2        
TARGET_CHUNKS = 25
TIMEOUT = 120

def normalize_pdf_text(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    paragraphs = []
    buf = ""

    for line in lines:
        if re.match(r"^[A-Z][^a-z]{3,}$", line):
            if buf:
                paragraphs.append(buf.strip())
                buf = ""
            paragraphs.append(line)
        elif buf.endswith((".", "!", "?")):
            paragraphs.append(buf.strip())
            buf = line
        else:
            buf = f"{buf} {line}" if buf else line

    if buf:
        paragraphs.append(buf.strip())

    return paragraphs

def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512
        }
    }

    r = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

class RAGDocumentProcessor:

    def extract_text(self, path: str) -> str:
        doc = fitz.open(path)
        pages = [p.get_text() for p in doc]
        doc.close()
        return "\n".join(pages)

    def analyze(self, paragraphs: List[str]) -> Dict:
        full = "\n".join(paragraphs)
        return {
            "chars": len(full),
            "words": len(full.split()),
            "paragraphs": len(paragraphs),
        }

    def chunk(self, paragraphs: List[str]) -> List[str]:
        if not paragraphs:
            return []

        total_chars = sum(len(p) for p in paragraphs)
        target_size = max(total_chars // TARGET_CHUNKS, 1500)

        chunks, buf = [], ""
        for p in paragraphs:
            if len(buf) + len(p) > target_size and buf:
                chunks.append(buf.strip())
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p

        if buf:
            chunks.append(buf.strip())

        return chunks

    def prompt(self, text: str) -> str:
        return f"""
            You are a text processor.

            Task:
            Split the following content into clean, readable segments.

            Rules:
            - Preserve the original wording as much as possible
            - Do NOT summarize
            - Do NOT paraphrase
            - Do NOT add new information
            - Do NOT remove meaningful details
            - Fix only obvious line-break issues if needed
            - Keep paragraphs intact

            Output:
            Return the processed text exactly as segmented text.
            No bullet points.
            No headings.
            No commentary.

            Content:
            {text[:MAX_LLM_CHARS]}
            """


    def extract_keypoints(self, idx: int, chunk: str) -> Dict:
        try:
            response = ollama_generate(self.prompt(chunk))
            return {
                "id": idx,
                "keypoints": response
            }
        except Exception as e:
            return {
                "id": idx,
                "keypoints": f"ERROR: {str(e)}"
            }

    def process(self, pdf_path: str):
        raw_text = self.extract_text(pdf_path)
        paragraphs = normalize_pdf_text(raw_text)

        analysis = self.analyze(paragraphs)
        chunks = self.chunk(paragraphs)

        if not chunks:
            raise RuntimeError("Chunking failed")

        results = []
        with ThreadPoolExecutor(MAX_WORKERS) as pool:
            futures = [
                pool.submit(self.extract_keypoints, i + 1, c)
                for i, c in enumerate(chunks)
            ]

            for f in tqdm(as_completed(futures), total=len(futures)):
                results.append(f.result())

        results.sort(key=lambda x: x["id"])
        return results, analysis

    def save(self, results, analysis, source):
        data = {
            "metadata": {
                "source": source,
                "model": OLLAMA_MODEL,
                "created_at": time.time()
            },
            "document_analysis": analysis,
            "contextual_keypoints": [
                {
                    "section_number": r["id"],
                    "title": f"Section {r['id']}",
                    "location": f"Chunk {r['id']}",
                    "theme": "Narrative",
                    "keypoints": r["keypoints"],
                    "keypoints_length": len(r["keypoints"])
                }
                for r in results
                if r["keypoints"]
                and not r["keypoints"].startswith("ERROR")
                and len(r["keypoints"]) > 50
            ]
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    pdf = "./database/sample_document.pdf"
    processor = RAGDocumentProcessor()
    results, analysis = processor.process(pdf)
    processor.save(results, analysis, pdf)

    print("✅ Extraction complete")
    print("Chunks:", len(results))
    print("Paragraphs:", analysis["paragraphs"])
    print("Saved to:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
