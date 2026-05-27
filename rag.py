import os
import re
import glob
import subprocess
import json
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# ── Step 1: Detect best subtitle language ────────────────────────────────────

def detect_best_source_language(url: str):
    """Returns (lang_code, reason) for the best available subtitle."""
    cmd = f'yt-dlp --skip-download --print-json "{url}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        return None, "Error: Could not fetch metadata."

    metadata = json.loads(result.stdout)
    manual_keys = list(metadata.get("subtitles", {}).keys())
    auto_keys   = list(metadata.get("automatic_captions", {}).keys())

    if "en" in manual_keys:
        return "en", "Manual English"
    if manual_keys:
        return manual_keys[0], f"Manual {manual_keys[0]}"

    orig = [k for k in auto_keys if k.endswith("-orig")]
    if orig:
        return orig[0], f"Original Audio ({orig[0]})"
    if "en" in auto_keys:
        return "en", "Auto-generated English"
    if auto_keys:
        return auto_keys[0], f"Auto-generated {auto_keys[0]}"

    return None, "No subtitles found."


# ── Step 2: Download transcript as .srt ──────────────────────────────────────

def download_transcript(video_url: str, lang_code: str, reason: str, output_name="final_transcript"):
    for f in glob.glob(f"{output_name}*"):
        try:
            os.remove(f)
        except Exception:
            pass

    if "Manual" in reason:
        sub_flag = "--write-subs"
    else:
        sub_flag = "--write-auto-subs"

    cmd = (
        f'yt-dlp --skip-download {sub_flag} '
        f'--sub-langs "{lang_code}" '       
        f'-o "{output_name}" "{video_url}"'
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Look for .vtt or .srt
    for ext in ["*.srt", "*.vtt"]:
        matches = glob.glob(f"{output_name}{ext}")
        if matches:
            return matches[0]

    return None

### new step
def parse_transcript_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if filepath.endswith(".vtt"):
        lines = content.split("\n")
        cleaned = []
        skip_next = False

        for line in lines:
            # Skip WEBVTT header and metadata lines
            if line.startswith("WEBVTT") or line.startswith("Kind:") or \
               line.startswith("Language:") or line.startswith("NOTE") or \
               line.startswith("STYLE") or line.startswith("REGION"):
                skip_next = False
                continue

            # Convert VTT timestamps to SRT style
            if "-->" in line:
                line = line.replace(".", ",")
                line = line.split(" align:")[0].split(" position:")[0].strip()
                cleaned.append(line)
                continue

            # Remove VTT inline tags like <00:00:01.234><c>text</c>
            line = re.sub(r"<[^>]+>", "", line).strip()
            cleaned.append(line)

        return "\n".join(cleaned)

    return content
# ── Step 3: Parse SRT and merge into Documents ───────────────────────────────

def build_documents(filepath: str, source_lang_code: str,
                    video_url: str, batch_size=100,
                    groq_api_key: str = None):
    """
    Reads an SRT file, merges every `batch_size` subtitle blocks into one
    LangChain Document. Translates non-English content with Groq if needed.
    """
    content = parse_transcript_file(filepath)

    blocks = re.split(r"\n\s*\n", content.strip())
    is_english = source_lang_code.startswith("en")
    final_docs = []


    llm_translator = None
    if not is_english and groq_api_key:
        llm_translator = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            api_key=groq_api_key,
        )

    for i in range(0, len(blocks), batch_size):
        batch = blocks[i: i + batch_size]
        texts, timestamps = [], []

        for block in batch:
            lines = block.split("\n")
            lines = [l for l in lines if l.strip()]  # remove empty lines
            
            if len(lines) < 2:
                continue

            # Check if first line is a timestamp
            if "-->" in lines[0]:
                timestamps.append(lines[0].split(" --> ")[0])
                texts.append(" ".join(lines[1:]).strip())
            # SRT format: block number, then timestamp, then text
            elif len(lines) >= 3 and "-->" in lines[1]:
                timestamps.append(lines[1].split(" --> ")[0])
                texts.append(" ".join(lines[2:]).strip())

        if not texts:
            continue

        if is_english:
            translated = texts
        else:
            try:
                translated = _translate_with_llm(texts, source_lang_code, llm_translator)
            except Exception:
                translated = texts

        merged_text = " ".join(translated)
        start_ts = timestamps[0].split(" --> ")[0] if timestamps else "00:00:00"

        final_docs.append(Document(
            page_content=merged_text,
            metadata={"timestamp": start_ts, "source": video_url},
        ))


    return final_docs


def _translate_with_llm(text_list, source_lang, llm):
    formatted = "\n".join([f"{i}: {t}" for i, t in enumerate(text_list)])
    prompt = (
        f"You are a professional translator. Translate the following subtitles "
        f"from {source_lang} to English. Maintain exact line numbering. "
        f"No commentary.\n\nSubtitles:\n{formatted}"
    )
    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n")
    return [re.sub(r"^\d+:\s*", "", l) for l in lines]


# ── Step 4: Embed and build FAISS vector store ────────────────────────────────

def build_vector_store(documents: list, index_path="faiss_youtube_index"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(index_path)
    return vector_db, embeddings, len(chunks)


# ── Step 5: Build the RAG chain ───────────────────────────────────────────────

def build_rag_chain(vector_db, embeddings, groq_api_key: str):
    # Retriever
    single_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Re-ranker
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=15)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=single_retriever,
    )

    # LLM
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        api_key=groq_api_key,
    )

    # Prompt
    system_prompt = (
        "You are an expert assistant for YouTube video transcripts. "
        "Use the retrieved context below to answer the question. "
        "Mention the timestamp of information when possible. "
        "If you don't know the answer, say so.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, qa_chain)
    return rag_chain


# ── Step 6: Ask a question ────────────────────────────────────────────────────

def ask_question(rag_chain, question: str) -> str:
    response = rag_chain.invoke({"input": question})
    return response["answer"]
