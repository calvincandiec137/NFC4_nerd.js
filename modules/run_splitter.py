import json
import re
import fitz, requests, os #type:ignore
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm #type:ignore

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
        print(f"Successfully extracted text from {os.path.basename(pdf_path)}")
        return full_text
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return ""

def contextually_split_text(text: str) -> list:
    pattern = r'(?m)(?=^\s*(?:Chapter\s+\d+|Section\s+[A-Z\d\.]+|[IVXLCDM]+\.|[A-Z\d]+\.|\b(?:Introduction|Abstract|Conclusion|Results|Methodology|Discussion|References)\b))'
    chunks = re.split(pattern, text)
    valid_chunks = [chunk.strip() for chunk in chunks if chunk and len(chunk.strip()) > 150]
    print(f"Split text into {len(valid_chunks)} contextual chunks.")
    return valid_chunks

def summarize_chunk_with_ollama(chunk: str, model_name: str = "phi3") -> tuple[str, str]:

    first_line = chunk.split('\n', 1)[0].strip()
    section_title = first_line if len(first_line) < 100 else "Untitled Section"

    prompt = f"""
    Concisely summarize the following document section. Focus on the key arguments, findings, and conclusions.
    Do not add any preamble like 'Here is the summary'. Just provide the raw summary text.

    ---
    TEXT SECTION:
    {chunk}
    ---

    SUMMARY:
    """
    try:
        response = requests.post("http://localhost:11434/api/generate",
                                 json={"model": model_name, "prompt": prompt, "stream": False},
                                 timeout=120)
        response.raise_for_status()
        summary = response.json().get('response', 'Error: No response text found.').strip()
        return section_title, summary
    except Exception as e:
        return section_title, f"ERROR processing this chunk: {e}"

def process_document(pdf_path: str) -> list:
    """
    Orchestrates the document processing pipeline using parallel summarization.
    """
    print("--- Document Processing Pipeline Initialized ---")
    document_text = extract_text_from_pdf(pdf_path)
    if not document_text:
        return []
    
    text_chunks = contextually_split_text(document_text)
    if not text_chunks:
        print("No valid chunks found to process.")
        return []

    all_summaries = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        print(f"\n--- Contacting Ollama to Summarize {len(text_chunks)} Sections in Parallel ---")
        
        futures = [executor.submit(summarize_chunk_with_ollama, chunk) for chunk in text_chunks]

        for future in tqdm(futures, total=len(text_chunks), desc="Summarizing Chunks"):
            try:
                section_title, summary = future.result()
                all_summaries.append({"section_title": section_title, "summary": summary})
            except Exception as e:
                print(f"A task generated an exception: {e}")

    return all_summaries


if __name__ == "__main__":
    summaries = process_document("sample_document.pdf")
    with open("summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4, ensure_ascii=False)
    print("✅ Summaries saved to summaries.json")

