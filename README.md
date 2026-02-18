# YouTube RAG Assistant

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue)](#)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-black)](#)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)](#)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-ff4b4b)](#)
[![Architecture](https://img.shields.io/badge/Architecture-RAG-success)](#)

Updated: 2026-02-18

AI application that converts YouTube videos into concise summaries and context-aware Q&A using Retrieval-Augmented Generation (RAG).

## What It Does

- Extracts transcript from a YouTube video
- Splits transcript into semantic chunks
- Builds vector index with FAISS
- Retrieves top relevant chunks for each question
- Generates grounded answers and clean summaries with OpenAI

## Tech Stack

- Python
- OpenAI API (`chat.completions` + embeddings)
- FAISS (`faiss-cpu`)
- Gradio
- `youtube-transcript-api`

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="your_openai_key"
python3 ytbot.py
```

## Resume Highlights

- Built an end-to-end RAG pipeline for long-form video understanding.
- Implemented semantic retrieval with FAISS for grounded question answering.
- Delivered a production-style UI with transcript diagnostics, caching, and port fallback.

## License

MIT
