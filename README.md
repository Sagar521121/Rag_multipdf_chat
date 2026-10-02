# Multi-PDF Research Assistant

A Streamlit retrieval-augmented generation (RAG) app for asking grounded questions across multiple PDF documents. It builds a local, persistent Chroma index and uses Gemini only after relevant evidence has been retrieved.

## Live Demo

[Launch the Multi-PDF Research Assistant](https://ragmultipdfchat-gr2krdsjazsyt7wx2umjaq.streamlit.app/)

## Application Preview

![Multi-PDF Research Assistant screenshot](assets/streamlit-demo.png)

## Features

- Upload several PDFs and explicitly build the index when ready.
- Extract page-aware text with an `unstructured` fast path and a `pypdf` fallback.
- Create normalized BGE embeddings and store them in a local Chroma collection.
- Answer only from retrieved passages, with source, page, and chunk evidence visible in the UI.
- Reject unsupported questions instead of fabricating an answer.
- Replace or remove individual indexed documents without rebuilding everything.

## Architecture

```text
PDF upload
  -> text extraction and page filtering
  -> chunking
  -> BGE embeddings
  -> local Chroma vector store
  -> relevance check
  -> Gemini grounded answer with citations
```

## Project structure

```text
.
|-- app.py                 # Streamlit user interface
|-- rag/                   # RAG pipeline modules and executable tests
|   |-- ingestion.py        # PDF extraction and chunking
|   |-- embeddings.py       # BGE embedding model
|   |-- vectorstore.py      # Persistent Chroma operations
|   |-- retrieval.py        # Similarity search
|   |-- relevance.py        # Retrieval threshold checks
|   |-- augmentation.py     # Context construction
|   `-- generation.py       # Gemini response generation
|-- data/PDFs/             # Local PDFs only; excluded from Git
|-- requirements.txt
`-- .env.example           # Safe environment-variable template
```

## Prerequisites

- Python 3.10 or later
- A Gemini API key

## Setup

1. Clone the repository and enter the project folder.

   ```powershell
   git clone https://github.com/<your-username>/<your-repository>.git
   cd <your-repository>
   ```

2. Create and activate a virtual environment.

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies.

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create your private configuration file from the template.

   ```powershell
   Copy-Item .env.example .env
   ```

5. Add your Gemini key to `.env`, then run the application.

   ```powershell
   streamlit run app.py
   ```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Gemini API key used only at runtime. |
| `GEMINI_MODEL` | No | Overrides the application's default Gemini model. |
| `RAG_DISTANCE_THRESHOLD` | No | Chroma distance cutoff for relevance; default: `1.10`. |

Never commit `.env`, `.streamlit/secrets.toml`, `chroma_db/`, or your own PDFs. They are excluded by `.gitignore`. If a key was ever committed to any repository, revoke it in Google AI Studio and create a new one before pushing.

## How to use

1. Start the app and upload one or more PDFs.
2. Select **Build index** to extract, chunk, embed, and store the documents.
3. Ask a question about the indexed documents.
4. Open **Inspect retrieved evidence** to review the passages used in the answer.

The app stores its index locally in `chroma_db/`; uploaded PDFs are handled as temporary files during indexing. Re-indexing a document replaces its existing vectors.

## Tests

Run the checks individually from the project root:

```powershell
python -m rag.test_ingestion
python -m rag.test_embeddings
python -m rag.test_vectorstore
python -m rag.test_retrieval
python -m rag.test_relevance
python -m rag.test_augmentation
python -m rag.test_generation
python -m rag.test_end_to_end
```

The embedding and end-to-end tests may download the BGE model on first run. The generation test uses a fake Gemini response and does not call the API.

## Publishing checklist

Before pushing, review the staged files with `git status` and `git diff --cached`. Only source code, documentation, configuration templates, and intentionally public assets should be staged.

This repository does not include a license. Add one only after choosing the terms under which you want others to use your code.
