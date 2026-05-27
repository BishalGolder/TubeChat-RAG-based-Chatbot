# 📺 TubeChat — Chat with Any YouTube Video

### High-Speed RAG Pipeline with Local CPU Embeddings, CrossEncoder Reranking & Groq Inference

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/Framework-LangChain-00A67E?logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-6A5ACD)
![Groq](https://img.shields.io/badge/Inference-Groq-orange?logo=groq&logoColor=white)

</p>

> TubeChat is an advanced Retrieval-Augmented Generation (RAG) system that converts long-form YouTube videos into an interactive conversational knowledge base with timestamp-aware responses.

---

# 🚀 Live Demo

🔗 **Web Deployment:**  
https://tubechat-rag.streamlit.app/

---

# ✨ Features

- 🎯 Ask questions about any public YouTube video
- ⏱️ Timestamp-grounded contextual responses
- 🌍 Multilingual subtitle processing pipeline
- 🧠 Hybrid Retrieval + CrossEncoder reranking
- ⚡ Sub-second response generation using Groq
- 💻 Fully CPU-compatible local embedding pipeline
- 📚 Supports long podcasts, lectures, interviews & tutorials

---

# 🏗️ System Architecture

```text
       ┌────────────────────────┐
       │      YouTube URL       │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   Language Detection   │
       │ [yt-dlp Metadata Scan] │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  Transcript Ingestion  │
       │ [Subtitle Extraction]  │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ VTT Parsing & Cleaning │
       │ [Regex + Time Sync]    │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Recursive Chunking     │
       │ Chunk Size: 1500       │
       │ Overlap: 200           │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Local CPU Embeddings   │
       │ all-MiniLM-L6-v2       │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ FAISS Vector Store     │
       └───────────┬────────────┘
                   │
         🚀 USER QUERY
                   │
                   ▼
       ┌────────────────────────┐
       │ Similarity Search      │
       │ KNN Retrieval          │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ CrossEncoder Reranker  │
       │ BAAI/bge-reranker-base │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Groq Llama 3.3 70B     │
       │ Response Generation    │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Timestamped Output     │
       └────────────────────────┘
```

---

# 🔥 Key Technical Highlights

## 🌐 Multilingual Processing

TubeChat automatically detects subtitle language streams and supports multilingual transcript ingestion.

If subtitles are non-English:
- Transcripts are translated through Groq
- Clean translated chunks are embedded locally
- Retrieval remains semantically accurate

---

## 🎯 Two-Stage Retrieval Pipeline

### Stage 1 — Vector Similarity Search
Fast semantic retrieval using FAISS vector indexing.

### Stage 2 — CrossEncoder Reranking
Initial candidates are reranked using:

```python
BAAI/bge-reranker-base
```

This significantly improves contextual relevance over naive embedding similarity alone.

---

## ⏱️ Timestamp-Aware Responses

Every generated answer links back to the original video timeline for:
- Fast verification
- Context navigation
- Better user trust

---

## 💻 Lightweight CPU Architecture

The embedding pipeline is optimized for:
- Consumer laptops
- CPU-only execution
- No CUDA dependency
- Low memory overhead

Tested successfully on:
- Intel Core i5 11th Gen
- Windows 11
- No GPU acceleration

---

# 📊 Performance Benchmarks

### Test Environment

| Component | Specification |
|---|---|
| CPU | Intel Core i5-1135G7 |
| GPU | None |
| RAM | 8GB |
| OS | Windows 11 |

---

| Operation | Time |
|---|---|
| Subtitle Download | ~5 Seconds |
| Transcript Cleaning | ~2 Seconds |
| Embedding + Vectorization | ~8–10 Minutes |
| Query Response | Sub-Second |

---

# 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend UI | Streamlit |
| RAG Orchestration | LangChain |
| Transcript Extraction | yt-dlp |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Reranking | BAAI/bge-reranker-base |
| Inference Engine | Groq Llama 3.3 70B |

---

# 📦 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/TubeChat.git
cd TubeChat
```

---

## 2️⃣ Create Virtual Environment

### Using uv (Recommended)

```bash
uv venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file inside the root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from:

https://console.groq.com/

---

# ▶️ Run Locally

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 📁 Project Structure

```text
TubeChat/
│
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
│
├── app.py
├── rag.py
├── requirements.txt
├── .env
└── README.md
```

---

# 🧠 Engineering Insights

Building TubeChat provided deep practical exposure to:

- Advanced RAG architecture design
- CrossEncoder reranking systems
- Transcript sanitization pipelines
- Vector similarity optimization
- Chunking strategy tradeoffs
- CPU-efficient embedding workflows
- Long-context retrieval engineering

---

# 🗺️ Future Improvements

- [ ] Persistent FAISS disk caching
- [ ] One-click video summarization
- [ ] Real-time embedding progress tracking
- [ ] Multi-video indexing support
- [ ] Playlist-wide semantic querying
- [ ] Citation highlighting in responses

---

# 🤝 Contributing

Pull requests, ideas, and improvements are welcome.

Feel free to fork the project and experiment with new retrieval strategies or inference backends.

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful, consider starring the repository to support future development.

```
