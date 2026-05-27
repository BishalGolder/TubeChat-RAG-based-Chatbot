import streamlit as st
import os
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

from rag import (
    detect_best_source_language,
    download_transcript,
    build_documents,
    build_vector_store,
    build_rag_chain,
    ask_question,
)

# Page config - Clean, Wide layout
st.set_page_config(
    page_title="TubeChat — AI Video Assistant",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Consumer-Grade CSS Injection ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Background and Typography settings */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Premium Red-Orange-Yellow Gradient Logo Typography */
    .brand-logo {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ff3333 0%, #ff8800 50%, #ffcc00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        display: inline-block;
    }
    
    /* Elegant Custom Card Components */
    .feature-card {
        background: linear-gradient(135deg, #1e2640 0%, #111525 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #2d385e;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: #ff8800;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    
    /* Sidebar Overrides for high-contrast branding */
    section[data-testid="stSidebar"] {
        background-color: #111525 !important;
        border-right: 1px solid #1e2640;
    }
    
    /* Main Landing Page Header Typography */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ff3333 0%, #ff8800 50%, #ffcc00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #808dad;
        margin-top: 5px;
        margin-bottom: 40px;
    }
    
    /* Custom style override for the primary button to blend with logo colors */
    div.stButton > button[type="submit"] {
        background: linear-gradient(45deg, #ff3333 0%, #ff8800 100%) !important;
        border: none !important;
        color: white !important;
    }
    div.stButton > button[type="submit"]:hover {
        background: linear-gradient(45deg, #ff5555 0%, #ffa422 100%) !important;
        box-shadow: 0 0 15px rgba(255, 136, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# Helpers
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")

# ── Session state defaults ────────────────────────────────────────────────────
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "processed" not in st.session_state:
    st.session_state.processed = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Custom Sidebar Branding Layout using the Red-Orange-Yellow Gradient
    st.markdown('<div class="brand-logo">▶️ TubeChat</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #808dad; font-size: 0.9rem; margin-top: 4px;">Chat with any YouTube video</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # API key input (shown only if not set in env/secrets)
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at console.groq.com",
        )
        st.divider()

    # YouTube URL input
    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        value=st.session_state.video_url,
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    process_btn = st.button("⚡ Process Video", use_container_width=True, type="primary")

    # Process video when button clicked
    if process_btn:
        if not api_key:
            st.error("Please enter your Groq API key.")
        elif not video_url.strip():
            st.error("Please enter a YouTube URL.")
        else:
            st.session_state.video_url = video_url.strip()
            st.session_state.chat_history = []
            st.session_state.processed = False
            st.session_state.rag_chain = None

            try:
                # Step 1: Detect language
                with st.status("🔍 Detecting subtitle language...", expanded=True) as status:
                    lang_code, reason = detect_best_source_language(video_url.strip())
                    if not lang_code:
                        st.error(f"Could not find subtitles: {reason}")
                        st.stop()
                    st.write(f"✅ Language detected: **{lang_code}** ({reason})")

                    # Step 2: Download transcript
                    status.update(label="📥 Downloading transcript...")
                    srt_file = download_transcript(video_url.strip(), lang_code, reason)
                    if not srt_file:
                        st.error("Failed to download transcript. Check if the video has subtitles.")
                        st.stop()
                    st.write(f"✅ Transcript downloaded: `{srt_file}`")

                    # Step 3: Build documents
                    status.update(label="📄 Processing transcript...")
                    documents = build_documents(
                        srt_file, lang_code, video_url.strip(),
                        groq_api_key=api_key
                    )
                    st.write(f"✅ Created **{len(documents)}** document chunks")

                    # Step 4: Embed & index
                    status.update(label="🧠 Building vector store (this may take a minute)...")
                    st.write(f"✅ Created **{len(documents)}** document chunks")
                    print(f"Documents created: {len(documents)}")
                    if documents:
                        print(f"First doc content: {documents[0].page_content[:200]}")
                        print(f"First doc metadata: {documents[0].metadata}")
                    
                    vector_db, embeddings, num_chunks = build_vector_store(documents)

                    # Step 5: Build RAG chain
                    status.update(label="⚙️ Setting up RAG chain...")
                    st.session_state.rag_chain = build_rag_chain(
                        vector_db, embeddings, api_key
                    )
                    st.session_state.processed = True
                    status.update(label="✅ Ready! Start chatting →", state="complete")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    # Status indicator
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.processed:
        st.success("🎯 Video Loaded & Indexed!")
        
        # Extracted pure ID or fallback to clean string for aesthetics
        clean_url = st.session_state.video_url
        if "watch?v=" in clean_url:
            clean_url = clean_url.split("watch?v=")[-1][:11]
        elif "youtu.be/" in clean_url:
            clean_url = clean_url.split("youtu.be/")[-1][:11]
            
        st.caption(f"Active ID: `{clean_url}`")
        if st.button("🔄 Reset Environment", use_container_width=True):
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.session_state.processed = False
            st.session_state.video_url = ""
            st.rerun()
    else:
        st.info("Setup required: Paste a URL & contextually parse it to begin.")

    st.markdown("<br><hr style='border-color: #1e2640;'>", unsafe_allow_html=True)
    st.caption("Engineered via LangChain · Groq · FAISS")


# ── Main chat area ────────────────────────────────────────────────────────────

if not st.session_state.processed:
    # Consumer Landing / Dashboard Welcome Screen
    st.markdown('<h1 class="hero-title">Talk to your videos.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Drop a link, index the transcript, and search deep context instantly.</p>', unsafe_allow_html=True)
    
    # Clean Grid Layout for Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h4 style="color:white; margin:0px 0px 8px 0px;">Instant Synthesis</h4>
            <p style="color:#808dad; font-size:0.9rem; margin:0;">Transcripts are parsed, split, and embedded into local semantic indexes within seconds.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌐</div>
            <h4 style="color:white; margin:0px 0px 8px 0px;">Global Translation</h4>
            <p style="color:#808dad; font-size:0.9rem; margin:0;">Native structural handling for cross-language querying and automated deep translations.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h4 style="color:white; margin:0px 0px 8px 0px;">Llama 3.3 Accuracy</h4>
            <p style="color:#808dad; font-size:0.9rem; margin:0;">Retrieval Augmented Generation backed by specialized high-performance Groq architecture.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Active Workspace Top-Bar
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.markdown('<h2 style="margin-bottom:0px;">💬 Chat Session</h2>', unsafe_allow_html=True)
        st.caption("Querying indexed vector chunks from your selected media.")
    with top_col2:
        # Subtle back-link styled to match the new warm accent palette
        st.markdown(f'<a href="{st.session_state.video_url}" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:6px; border-radius:6px; border:1px solid #ff8800; background:transparent; color:#ff8800; cursor:pointer; font-weight:600;">📺 Open Source Video</button></a>', unsafe_allow_html=True)
    
    st.divider()

    # Render existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask something about the video...")

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context indexes..."):
                try:
                    answer = ask_question(st.session_state.rag_chain, user_input)
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    err_msg = f"❌ Error: {e}"
                    st.error(err_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err_msg}
                    )