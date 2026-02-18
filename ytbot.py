import os
import re
import socket
from typing import Any, Dict, List, Optional, Tuple

import faiss
import gradio as gr
import numpy as np
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


# In-memory cache to avoid refetching/reindexing the same video repeatedly.
_CACHE: Dict[str, Dict[str, Any]] = {}


def find_open_port(start: int = 7860, end: int = 9000) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    # Fallback: ask OS for any free ephemeral port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_transcript(url: str) -> Optional[List[Dict[str, Any]]]:
    transcript, _ = get_transcript_with_reason(url)
    return transcript


def get_transcript_with_reason(url: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    video_id = get_video_id(url)
    if not video_id:
        return None, "Invalid YouTube URL."

    try:
        # youtube-transcript-api versions differ: prefer instance API, keep compatibility fallback.
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)  # type: ignore[attr-defined]
        else:
            transcript_list = YouTubeTranscriptApi().list(video_id)
        chosen = None
        for t in transcript_list:
            if t.language_code == "en":
                if not t.is_generated:
                    chosen = t
                    break
                if chosen is None:
                    chosen = t

        if chosen is None:
            # Fallback: use any transcript that can be translated to English.
            for t in transcript_list:
                if getattr(t, "is_translatable", False):
                    chosen = t.translate("en")
                    break

        if chosen is None:
            return None, "No transcript found for this video."

        fetched = chosen.fetch()
        # Some versions return objects with to_raw_data(); others are already dict-like.
        if hasattr(fetched, "to_raw_data"):
            return fetched.to_raw_data(), None
        return [dict(item) for item in fetched], None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def process(transcript: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in transcript:
        text = item.get("text")
        start = item.get("start")
        if text is None or start is None:
            continue
        lines.append(f"Text: {text} Start: {start}")
    return "\n".join(lines)


def chunk_transcript(processed_transcript: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    if not processed_transcript:
        return []
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(processed_transcript):
        chunks.append(processed_transcript[start : start + chunk_size])
        start += step
    return chunks


def setup_openai() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    response = client.embeddings.create(model=model, input=texts)
    vectors = np.array([d.embedding for d in response.data], dtype=np.float32)

    # Normalize to use cosine similarity via inner product.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return vectors


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("No vectors available to build FAISS index.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve_context(client: OpenAI, query: str, faiss_index: faiss.IndexFlatIP, chunks: List[str], k: int = 7) -> List[str]:
    query_vec = embed_texts(client, [query])
    if query_vec.size == 0:
        return []

    top_k = min(k, len(chunks))
    _scores, indices = faiss_index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def summarize_with_openai(client: OpenAI, transcript: str) -> str:
    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize YouTube transcripts. Return one concise paragraph, "
                    "ignore timestamps, and focus on spoken factual content."
                ),
            },
            {"role": "user", "content": transcript},
        ],
    )
    return completion.choices[0].message.content.strip()


def answer_with_openai(client: OpenAI, context: str, question: str) -> str:
    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. If context is insufficient, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
    )
    return completion.choices[0].message.content.strip()


def ensure_video_ready(video_url: str) -> Dict[str, Any]:
    if not video_url:
        return {"error": "Please provide a YouTube URL."}

    video_id = get_video_id(video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL. Expected a watch URL or youtu.be URL."}

    if video_id in _CACHE:
        return _CACHE[video_id]

    transcript, transcript_error = get_transcript_with_reason(video_url)
    if not transcript:
        return {"error": transcript_error or "No transcript found for this video."}

    client = setup_openai()
    if client is None:
        return {"error": "Missing OpenAI credentials. Set OPENAI_API_KEY in your environment."}

    processed_transcript = process(transcript)
    chunks = chunk_transcript(processed_transcript)
    if not chunks:
        return {"error": "Transcript is empty after processing."}

    vectors = embed_texts(client, chunks)
    try:
        faiss_index = build_faiss_index(vectors)
    except ValueError as e:
        return {"error": str(e)}

    data = {
        "video_id": video_id,
        "processed_transcript": processed_transcript,
        "chunks": chunks,
        "faiss_index": faiss_index,
        "client": client,
    }
    _CACHE[video_id] = data
    return data


def summarize_video(video_url: str) -> str:
    data = ensure_video_ready(video_url)
    if "error" in data:
        return data["error"]
    return summarize_with_openai(data["client"], data["processed_transcript"])


def answer_question(video_url: str, user_question: str) -> str:
    if not user_question:
        return "Please enter a question."

    data = ensure_video_ready(video_url)
    if "error" in data:
        return data["error"]

    relevant_chunks = retrieve_context(
        data["client"],
        user_question,
        data["faiss_index"],
        data["chunks"],
        k=7,
    )
    context_text = "\n\n".join(relevant_chunks)
    return answer_with_openai(data["client"], context_text, user_question)


def fetch_status(video_url: str) -> str:
    data = ensure_video_ready(video_url)
    if "error" in data:
        return data["error"]

    formatter = TextFormatter()
    transcript, transcript_error = get_transcript_with_reason(video_url)
    transcript = transcript or []
    if not transcript:
        if transcript_error:
            return f"Transcript unavailable: {transcript_error}"
        return "Ready (cached), but transcript preview unavailable."

    preview = formatter.format_transcript(transcript[:3]).strip().replace("\n", " ")
    return f"Transcript fetched and indexed. Preview: {preview[:160]}..."


with gr.Blocks() as interface:
    gr.Markdown("<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>")

    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube URL")
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    with gr.Row():
        fetch_btn = gr.Button("Fetch Transcript")
        summarize_btn = gr.Button("Summarize Video")

    summary_output = gr.Textbox(label="Video Summary", lines=7)

    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    question_btn = gr.Button("Ask a Question")
    answer_output = gr.Textbox(label="Answer", lines=7)

    fetch_btn.click(fetch_status, inputs=video_url, outputs=transcript_status)
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)


if __name__ == "__main__":
    preferred = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    port = find_open_port(start=preferred, end=9000)
    interface.launch(server_name="127.0.0.1", server_port=port)
