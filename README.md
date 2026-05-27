# 📺 TubeChat — Chat with Any YouTube Video

> A RAG-based AI chatbot that lets you have a conversation with any YouTube video using its transcript.

🔗 **Live Demo:** [TubeChat-rag-based-chatbot.streamlit.app](https://TubeChat-rag-based-chatbot.streamlit.app)

---

## What is TubeChat?

TubeChat lets you paste any YouTube URL and instantly start asking questions about the video content. Instead of scrubbing through a 1-hour podcast to find one insight, just ask — and get an answer with timestamps in seconds.

It was tested on a **1-hour+ Jensen Huang podcast** on an Intel Core i5 11th Gen (no GPU), processing the full transcript in ~10 minutes with near-instant answer generation.

---

## How It Works

```
YouTube URL
     │
     ▼
Language Detection        ← yt-dlp metadata scan
     │
     ▼
Transcript Download       ← .vtt subtitle file
     │
     ▼
VTT Parsing & Cleaning    ← removes tags, fixes timestamps
     │
     ▼
Text Chunking             ← RecursiveCharacterTextSplitter
     │                       chunk_size=1500, overlap=200
     ▼
Embedding                 ← all-MiniLM-L6-v2 (runs on CPU)
     │
     ▼
FAISS Vector Index        ← saved locally for fast retrieval
     │
     ▼
Question Asked
     │
     ▼
Similarity Search (k=3)
     │
     ▼
CrossEncoder Reranker     ← BAAI/bge-reranker-base, top_n=15
     │
     ▼
Groq LLM                  ← Llama 3.3 70B via cloud API
     │
     ▼
Answer with Timestamps
```

---

## Features

- **Multilingual support** — detects subtitle language automatically, translates non-English to English using Groq LLM before indexing
- **Manual + auto subtitles** — handles both manual and auto-generated YouTube subtitles
- **CrossEncoder reranking** — improves retrieval quality over naive similarity search
- **Timestamp-aware answers** — responses reference where in the video information comes from
- **CPU-friendly** — embedding model runs locally without a GPU
- **Clean chat UI** — built with Streamlit, persistent chat history per session

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Transcript Download | yt-dlp |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder) |
| LLM | Groq API — Llama 3.3 70B Versatile |
| Orchestration | LangChain |

---

## Performance

Tested on **Intel Core i5 11th Gen, no GPU, Windows 11**

| Stage | Time |
|-------|------|
| Transcript download | ~5 seconds |
| Embedding (1hr+ video) | ~8–10 minutes |
| Answer generation | Near-instant (Groq cloud) |

Processing is a one-time cost per video. Once indexed, all questions are answered in milliseconds.

---

## Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/TubeChat.git
cd TubeChat

# Create virtual environment
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
uv pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
TubeChat/
├── app.py                  # Streamlit UI
├── rag.py                  # RAG pipeline logic
├── requirements.txt
├── .env                    # API key (not committed)
└── .streamlit/
    ├── config.toml         # Streamlit config
    └── secrets.toml        # Streamlit Cloud secrets (not committed)
```

---

## What I Learned

This was my first end-to-end AI project, built from scratch in Google Colab over several days before adding a UI.

Key things I learned building this:
- Why naive similarity search isn't enough and how CrossEncoder reranking improves retrieval
- How chunking strategy affects answer quality — chunk size and overlap matter a lot
- How to handle multilingual content in a RAG pipeline
- The difference between manual and auto-generated subtitles and why it matters for download logic
- How to go from a Jupyter notebook to a deployed web application

---

## What I'd Improve Next

- Cache the FAISS index so re-loading the same video is instant
- Add a "Summarize this video" one-click button
- Show a progress bar during the embedding step
- Support loading multiple videos and asking across all of them
- Improve the repeated-text issue from VTT subtitle parsing

---

## License

MIT License — feel free to use, fork, and build on this.

---

*Built by [Your Name] · 3rd Year CS Student · First AI Project*
