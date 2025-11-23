# DoChat


https://github.com/user-attachments/assets/4f7fa70f-5eac-4f30-a9fd-b7d5a77e0873


**Demo Video:** https://youtu.be/smLLruXduPU

**Short Description**: DoChat is a lightweight Retrieval-Augmented Generation (RAG) utility for building embeddings from structured JSON documents, indexing them with FAISS, and querying the result to power context-aware chat or retrieval workflows.

**Features**
- **Embedding generation**: Builds dense text embeddings using `sentence-transformers`.
- **FAISS indexing**: Creates a FAISS index for fast similarity search.
- **Chunked processing**: Splits long text into configurable chunks (size & overlap).
- **Metadata tracking**: Saves per-chunk metadata to map results back to source sections.

**Repository Structure**
- **`database/`**: Input example files.
  - `sample_json.json` — Example source JSON used to build the index.
- **`embeddings/`**: Output folder created by the build process.
  - `vectors.npy` — Numpy array of saved embeddings.
  - `index.faiss` — FAISS index file.
  - `metadata.json` — JSON array containing metadata for each chunk.
- **`modules/`**: Main scripts and helpers.
  - `RAG.py` — Main script to build embeddings and the FAISS index.
  - `run_splitter.py` — (Splitter utility) splits long documents into chunks.
  - `response.py` — (Query utility) example flow to query the built index.
  - `last_query_result.json` — Example/result file used by the query flow.

- `demo.html` — Local demo page that auto-plays and loops the YouTube demo video (open in browser).

**Requirements**
- Python 3.8 or newer
- Recommended: create and use a virtual environment
- Key Python packages:
  - `sentence-transformers`
  - `faiss-cpu` (or `faiss` depending on your platform)
  - `numpy`
  - `tqdm`
  - `requests`

Create a `requirements.txt` with the following lines as a starting point:

```
sentence-transformers
numpy
tqdm
requests
faiss-cpu
```

**Installation**
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) If you prefer GPU FAISS, follow FAISS installation guide for your platform.

**Usage**

1) Build embeddings and FAISS index

```bash
python modules/RAG.py
```

- What this does:
  - Loads document data from `database/sample_json.json`.
  - Splits `keypoints` text into chunks using the configured `CHUNK_SIZE` and `CHUNK_OVERLAP`.
  - Generates embeddings in batch using a `sentence-transformers` model.
  - Saves `vectors.npy`, writes a FAISS index to `index.faiss`, and stores chunk-level `metadata.json`.

2) Splitter (if present)

```bash
python modules/run_splitter.py
```

- Use this to preprocess or re-chunk source text before building embeddings.

3) Query / Response

```bash
python modules/response.py
```

- This script demonstrates how to load the FAISS index and `metadata.json`, run a similarity search, and format the response. Results may be stored in `modules/last_query_result.json` depending on script logic.

**Configuration**
- Modify the following variables at the top of `modules/RAG.py` to suit your needs:
  - `CHUNK_SIZE` — default `1000`, length of each text chunk (characters)
  - `CHUNK_OVERLAP` — default `200`, overlap between adjacent chunks
  - `EMBED_MODEL` — currently `sentence-transformers/all-MiniLM-L6-v2`; replace to switch models
  - `DOCS_JSON_PATH` — path to your documents JSON (default: `./database/sample_json.json`)
  - `OUTPUT_DIR` — output directory for embeddings/index/metadata (default: `./embeddings`)

**Data Format**
The `sample_json.json` should contain a top-level `contextual_keypoints` array. Each element in the array should be an object with fields similar to:

```json
{
  "section_number": "1",
  "title": "Section Title",
  "keypoints": "Text or notes here...",
  "theme": "optional theme",
  "location": "optional location",
  "original_chunk_id": "optional id",
  "keypoints_length": 123
}
```

The `RAG.py` script looks up these keys when creating metadata entries. If a key is missing, sensible defaults are used.

**Outputs Explained**
- `embeddings/vectors.npy`: Numpy float32 matrix of shape (N, D) where N is number of chunks and D is embedding dimensionality.
- `embeddings/index.faiss`: FAISS index file that stores the index used for nearest neighbor search.
- `embeddings/metadata.json`: Array of metadata objects that correspond to rows in `vectors.npy` and entries in the FAISS index. Each metadata entry includes `doc_id`, `section_title`, `chunk_index`, and the original `text`.
- `modules/last_query_result.json`: (Optional) Stores last query's result structure for inspection.

**Examples**
- Build and check outputs:

```bash
python modules/RAG.py
ls -la embeddings
```

- Run a query example (modify `modules/response.py` to customize prompts or retrieval params):

```bash
python modules/response.py
cat modules/last_query_result.json
```

**Local Demo**

Open `demo.html` in your browser to run the looping YouTube demo. For a local HTTP server (recommended to avoid cross-origin issues), run:


Notes:
- The demo uses the YouTube IFrame Player API. The player is started muted to satisfy browser autoplay restrictions; click the "Unmute & Play" button to enable sound.
- GitHub's README preview will not render iframes — open `demo.html` directly in your browser or serve it as above.

**Troubleshooting**
- Empty or missing `database/sample_json.json`: Ensure path and JSON structure are correct. The builder will print `[❌] Failed to load JSON file` if the file cannot be read.
- Model download stalls: `sentence-transformers` will auto-download models. Confirm network access and enough disk space.
- FAISS errors: If `faiss` import fails, install `faiss-cpu` via pip or follow platform-specific instructions for `faiss`/GPU support.
- Memory issues generating many embeddings: Reduce batch size or process documents in smaller batches.

**License & Contact**
- **License**: MIT License — see the `LICENSE` file at the project root for the full text.
- **Copyright**: Copyright (c) 2025 `calvincandiec137`.

**Acknowledgements**
- `sentence-transformers` — easy-to-use sentence embedding models
- `FAISS` — fast similarity search
- `numpy`, `tqdm`, `requests`
