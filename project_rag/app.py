

import html
import io
import json
import os
import re
import time

import msgpack
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(
    page_title="LexAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
def resolve_endee_url() -> str:
    u = os.getenv("ENDEE_URL")
    if u: return u.rstrip("/")
    h = os.getenv("ENDEE_HOSTPORT")
    if h: return (h if h.startswith("http") else f"http://{h}").rstrip("/")
    return "http://localhost:8080"

ENDEE_URL        = resolve_endee_url()
INDEX_NAME       = os.getenv("INDEX_NAME", "LEGALDOC")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
def inject_theme():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #212121;
  --sidebar-bg: #171717;
  --surface:    #2f2f2f;
  --surface2:   #3a3a3a;
  --border:     #3a3a3a;
  --border2:    #4a4a4a;
  --text:       #ececec;
  --text2:      #b0b0b0;
  --muted:      #6a6a6a;
  --accent:     #10a37f;
  --accent2:    #1abf96;
  --accent-dim: rgba(16,163,127,0.12);
  --accent-bd:  rgba(16,163,127,0.30);
  --r:          12px;
  --rsm:        8px;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Hide ALL Streamlit chrome */
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stToolbar"],
#MainMenu, footer, header { display: none !important; }

/* Sidebar toggle button — keep visible */
[data-testid="collapsedControl"] {
  display: flex !important;
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="collapsedControl"] * { color: var(--text2) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Content area */
.block-container {
  max-width: 780px !important;
  margin: 0 auto !important;
  padding: 0 1.5rem 8rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* Text inputs */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--rsm) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.94rem !important;
  caret-color: var(--accent) !important;
  transition: border-color 0.15s, box-shadow 0.15s;
}
div[data-baseweb="textarea"] textarea:focus,
div[data-baseweb="input"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
  outline: none !important;
}
div[data-baseweb="textarea"] textarea::placeholder,
div[data-baseweb="input"] input::placeholder {
  color: var(--muted) !important; opacity: 1 !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface) !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: var(--r) !important;
  transition: border-color 0.15s;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--text2) !important; }

/* Buttons */
.stButton > button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
  border-radius: var(--rsm) !important;
  border: 1px solid var(--border2) !important;
  background: var(--surface) !important;
  color: var(--text2) !important;
  transition: all 0.12s !important;
  width: 100%;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-dim) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #fff !important;
  border-color: var(--accent) !important;
  font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent2) !important;
  border-color: var(--accent2) !important;
}

/* Alert */
[data-testid="stAlert"] {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--rsm) !important;
}
[data-testid="stAlert"] * { color: var(--text2) !important; }

/* Expander */
[data-testid="stExpander"] details {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--rsm) !important;
}
[data-testid="stExpander"] summary { color: var(--text2) !important; font-size: 0.82rem !important; }

/* Hide all spinners and loading indicators */
[data-testid="stStatus"]        { display: none !important; }
[data-testid="stSpinner"]       { display: none !important; }
[data-testid="stProgressBar"]   { display: none !important; }
.stSpinner                      { display: none !important; }
.element-container:has(.stSpinner) { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* ════════════════
   SIDEBAR
   ════════════════ */
.sb-brand {
  display: flex; align-items: center; gap: 10px;
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--border);
}
.sb-icon {
  width: 32px; height: 32px; flex-shrink: 0;
  background: var(--accent); border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px;
}
.sb-name { font-size: 0.95rem; font-weight: 700; color: var(--text); }
.sb-sub  { font-size: 0.62rem; color: var(--muted); margin-top: 1px; }
.sb-divider { border: none; border-top: 1px solid var(--border); margin: 8px 0; }
.sb-doc-pill {
  margin: 10px 10px 4px; padding: 10px 12px;
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: var(--rsm);
}
.sb-doc-name {
  font-size: 0.8rem; font-weight: 600; color: var(--text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.sb-doc-lang {
  display: inline-block; margin-top: 4px; font-size: 0.62rem; font-weight: 600;
  padding: 2px 7px; border-radius: 999px;
  background: var(--accent-dim); border: 1px solid var(--accent-bd); color: var(--accent);
}
.sb-nav-label {
  padding: 12px 14px 5px; font-size: 0.6rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted);
}

/* ════════════════
   PAGE HEADER
   ════════════════ */
.main-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 22px 0 14px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}
.app-name {
  font-size: 1.75rem; font-weight: 800; color: var(--text);
  display: flex; align-items: center; gap: 10px; letter-spacing: -0.5px;
}
.app-name-icon {
  font-size: 1.75rem; line-height: 1;
}
.doc-tag {
  display: flex; align-items: center; gap: 0;
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 999px; padding: 4px 12px; overflow: hidden;
}
.doc-tag-name {
  font-size: 0.72rem; color: var(--text2);
  max-width: 180px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
.doc-tag-sep  { font-size: 0.62rem; color: var(--border2); margin: 0 6px; }
.doc-tag-lang { font-size: 0.62rem; color: var(--muted); }

/* ════════════════
   WELCOME
   ════════════════ */
.welcome-wrap {
  text-align: center; padding: 44px 0 28px;
}
.welcome-icon {
  width: 64px; height: 64px; margin: 0 auto 18px;
  background: var(--accent-dim); border: 1.5px solid var(--accent-bd);
  border-radius: 16px; display: flex; align-items: center;
  justify-content: center; font-size: 30px;
}
.welcome-title { font-size: 1.85rem; font-weight: 800; color: var(--text); margin-bottom: 8px; letter-spacing: -0.5px; }
.welcome-desc  {
  font-size: 0.875rem; color: var(--text2); line-height: 1.65;
  max-width: 380px; margin: 0 auto 28px;
}


/* ════════════════
   CHAT MESSAGES
   ════════════════ */
.msg-group { opacity: 0; animation: msgIn 0.18s ease forwards; }
@keyframes msgIn {
  from { opacity: 0; transform: translateY(5px); }
  to   { opacity: 1; transform: translateY(0); }
}

.msg-row { display: flex; gap: 11px; padding: 14px 0; align-items: flex-start; }
.msg-row.user-row { justify-content: flex-end; }

.av {
  width: 28px; height: 28px; flex-shrink: 0; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; margin-top: 1px;
}
.av-lex  { background: var(--accent); color: #fff; }
.av-user { background: var(--surface2); border: 1px solid var(--border2); color: var(--text2); }

.bubble {
  font-size: 0.9rem; line-height: 1.75; word-break: break-word; max-width: 88%;
}
.bubble.user {
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: var(--r) var(--rsm) var(--r) var(--r);
  padding: 10px 15px; color: var(--text);
}
.bubble.ai {
  color: var(--text); padding: 2px 0; max-width: 92%;
}

/* Rendered markdown inside AI bubble */
.bubble.ai p  { margin-bottom: 0.6em; }
.bubble.ai p:last-child { margin-bottom: 0; }
.bubble.ai strong { font-weight: 600; color: var(--text); }
.bubble.ai em     { font-style: italic; color: var(--text2); }
.bubble.ai ul, .bubble.ai ol { padding-left: 1.3em; margin-bottom: 0.6em; }
.bubble.ai li     { margin-bottom: 0.25em; }
.bubble.ai br     { display: block; content: ""; margin-top: 0.3em; }

/* ════════════════
   CHAIN-LINK CITATION  (icon only, no number)
   ════════════════ */
.cite-link {
  display: inline-flex; align-items: center; justify-content: center;
  vertical-align: middle; margin: 0 2px;
  text-decoration: none !important; cursor: pointer;
  transition: opacity 0.12s;
}
.cite-link:hover { opacity: 0.65; }
.cite-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 17px; height: 17px;
  background: var(--surface2); border: 1px solid var(--border2);
  border-radius: 4px; flex-shrink: 0;
  transition: background 0.12s, border-color 0.12s;
}
.cite-link:hover .cite-icon { background: var(--accent-dim); border-color: var(--accent-bd); }
.cite-icon svg { width: 9px; height: 9px; display: block; }

/* ════════════════
   SOURCE CARDS
   ════════════════ */
.sources-panel { margin: 8px 0 4px 40px; }
.src-item {
  display: flex; gap: 10px; align-items: flex-start;
  padding: 8px 12px; margin-bottom: 5px;
  background: var(--surface); border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--rsm) var(--rsm) 0;
}
.src-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem; font-weight: 700; color: var(--accent);
  flex-shrink: 0; min-width: 16px; margin-top: 2px;
}
.src-snippet {
  font-size: 0.78rem; color: var(--text2); line-height: 1.5;
  display: -webkit-box; -webkit-line-clamp: 3;
  -webkit-box-orient: vertical; overflow: hidden;
}
.hl { background: rgba(16,163,127,0.18); border-radius: 2px; padding: 0 2px; color: var(--accent2); }
.msg-divider { border: none; border-top: 1px solid var(--border); margin: 2px 0; }

/* ════════════════
   SIMPLIFY CARDS
   ════════════════ */
.simp-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--rsm); padding: 14px 16px; margin-bottom: 8px;
}
.simp-type {
  font-size: 0.62rem; font-weight: 700; color: var(--accent);
  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;
}
.simp-orig {
  font-size: 0.75rem; color: var(--muted); font-style: italic;
  border-left: 2px solid var(--border2); padding-left: 8px;
  margin-bottom: 8px; line-height: 1.45;
}
.simp-plain { font-size: 0.875rem; color: var(--text); line-height: 1.68; }

.sec-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 18px 0 12px; border-bottom: 1px solid var(--border); margin-bottom: 18px;
}
.sec-title { font-size: 1rem; font-weight: 600; color: var(--text); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_state():
    for k, v in {
        "chat_history": [],
        "doc_text":     "",
        "doc_name":     "",
        "doc_lang":     "en",
        "simplified":   [],
        "view":         "home",
        "input_key":    0,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# ENDEE + EMBEDDING
# ─────────────────────────────────────────────
def endee_h(json_ct=False):
    h = {}
    if ENDEE_AUTH_TOKEN: h["Authorization"] = ENDEE_AUTH_TOKEN
    if json_ct: h["Content-Type"] = "application/json"
    return h or None

def is_endee_ok() -> bool:
    try:
        return requests.get(f"{ENDEE_URL}/api/v1/health", headers=endee_h(), timeout=2).ok
    except: return False

@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)

def create_index() -> bool:
    try:
        r = requests.post(f"{ENDEE_URL}/api/v1/index/create",
            json={"index_name": INDEX_NAME, "dim": 384, "space_type": "cosine",
                  "precision": "float32", "M": 16, "ef_con": 200},
            headers=endee_h(True), timeout=10)
        return r.ok
    except: return False

def delete_index():
    try:
        r = requests.delete(f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/delete",
                            headers=endee_h(), timeout=10)
        return r.status_code in (200, 404)
    except: return False

def chunk_text(text: str, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size) if words[i:i+size]]

def ingest_doc(text: str, model, doc_name: str):
    chunks = chunk_text(text)
    if not chunks: return
    embs = model.encode(chunks, normalize_embeddings=True)
    seed = f"{doc_name}-{int(time.time())}"
    vecs = [{"id": f"{seed}-{i}", "vector": e.astype(np.float32).tolist(),
              "meta": json.dumps({"doc": doc_name, "text": c})}
             for i, (c, e) in enumerate(zip(chunks, embs))]
    try:
        r = requests.post(f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
                          json=vecs, headers=endee_h(True), timeout=60)
        if not r.ok and r.status_code == 400 and "required files missing" in r.text.lower():
            delete_index(); create_index()
            requests.post(f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
                          json=vecs, headers=endee_h(True), timeout=60)
    except: pass

def parse_meta(mv):
    if isinstance(mv, bytes): mv = mv.decode("utf-8", errors="replace")
    if isinstance(mv, dict): return str(mv.get("doc","Doc")), str(mv.get("text",""))
    if isinstance(mv, str):
        try:
            p = json.loads(mv)
            if isinstance(p, dict): return str(p.get("doc","Doc")), str(p.get("text",""))
        except: return "Document", mv
    return "Document", str(mv)

def do_search(question: str, model, top_k=4):
    qe = model.encode([question])[0]
    try:
        r = requests.post(f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search",
                          json={"vector": qe.astype(np.float32).tolist(), "k": top_k},
                          headers=endee_h(True), timeout=20)
        if not r.ok: return []
        raw = msgpack.unpackb(r.content, raw=False)
        srcs = []
        for item in (raw if isinstance(raw, list) else []):
            if isinstance(item, dict):
                mv = item.get("meta",""); sc = item.get("score", item.get("distance"))
            elif isinstance(item, list) and len(item) > 2:
                sc = item[1]; mv = item[2]
            else: continue
            dn, tx = parse_meta(mv)
            srcs.append({"doc": dn, "text": tx, "score": sc})
        return srcs
    except: return []


# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────
def detect_lang(text: str) -> str:
    """Detect Kannada vs English."""
    return "kn" if sum(1 for c in text if '\u0C80' <= c <= '\u0CFF') > 50 else "en"

def detect_question_lang(question: str) -> str:
    """Detect the language the user typed their question in."""
    return "kn" if sum(1 for c in question if '\u0C80' <= c <= '\u0CFF') > 5 else "en"


# ─────────────────────────────────────────────
# PDF EXTRACTION — clean text
# ─────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages)
    except Exception:
        pass

    if not text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            pass

    if text:
        # Remove non-printable / garbage bytes; keep Kannada Unicode block
        text = re.sub(r'[^\x20-\x7E\u0C80-\u0CFF\u0900-\u097F\n\r\t]', ' ', text)
        text = re.sub(r'[ \t]{3,}', ' ', text)
        text = re.sub(r'\n{4,}', '\n\n', text)
        return text.strip()
    return text


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
def groq_ok() -> bool:
    return bool(GROQ_API_KEY and GROQ_API_KEY not in ("", "your_groq_api_key_here"))

def llm_call(prompt: str, temperature=0.2, max_tokens=1200) -> str:
    if not groq_ok():
        return "⚠️ GROQ_API_KEY not configured. Add it to your .env file."
    c = Groq(api_key=GROQ_API_KEY)
    r = c.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content


# Chain-link SVG — icon only, no number
CHAIN_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" '
    'stroke="#10a37f" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" '
    'stroke="#10a37f" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)

def make_cite_html(idx: int, sources: list) -> str:
    """Return a chain-link icon only — no superscript number."""
    if 1 <= idx <= len(sources):
        tip = html.escape(sources[idx-1].get("text","")[:90].replace('"',"'"))
        return (f'<a class="cite-link" href="#src-{idx}" title="{tip}">'
                f'<span class="cite-icon">{CHAIN_SVG}</span>'
                f'</a>')
    return ""


def markdown_to_html(text: str) -> str:
    """
    Convert markdown-style formatting to clean HTML.
    Handles **bold**, *italic*, numbered lists, bullet lists, line breaks.
    Strips raw ** that the LLM sometimes leaves in.
    """
    # Bold: **text** → <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text* → <em>text</em>  (single * not preceded by another *)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)

    lines = text.split('\n')
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            html_parts.append('')
            continue

        # Numbered list: "1. " or "1) "
        num_match = re.match(r'^(\d+)[.)]\s+(.*)', stripped)
        # Bullet list: "- " or "• "
        bul_match = re.match(r'^[-•]\s+(.*)', stripped)

        if num_match or bul_match:
            if not in_list:
                html_parts.append('<ul style="padding-left:1.4em;margin:0.4em 0">')
                in_list = True
            content = num_match.group(2) if num_match else bul_match.group(1)
            html_parts.append(f'<li style="margin-bottom:0.3em">{content}</li>')
        else:
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            html_parts.append(f'<p style="margin-bottom:0.5em">{stripped}</p>')

    if in_list:
        html_parts.append('</ul>')

    return '\n'.join(html_parts)


def render_answer(answer_text: str, sources: list) -> str:
    """
    1. Replace [N] citation markers with chain-link icons (no number).
    2. Convert markdown to clean HTML.
    """
    # Step 1: extract and replace citations before markdown conversion
    # to avoid markdown processing interfering with brackets
    cite_map = {}
    def store_cite(m):
        idx = int(m.group(1))
        placeholder = f"__CITE_{idx}__"
        cite_map[placeholder] = make_cite_html(idx, sources)
        return placeholder

    text = re.sub(r'\[(\d+)\]', store_cite, answer_text)

    # Step 2: convert markdown to HTML
    text = markdown_to_html(text)

    # Step 3: restore citations
    for placeholder, cite_html in cite_map.items():
        text = text.replace(html.escape(placeholder), cite_html)
        text = text.replace(placeholder, cite_html)

    return text


# ─────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────
def generate_answer(question: str, sources: list, q_lang: str) -> str:
    """
    Answer language = question language, regardless of document language.
    If question is in Kannada → answer in Kannada.
    If question is in English → answer in English.
    """
    if not sources:
        if q_lang == "kn":
            return "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ದಾಖಲೆಯಲ್ಲಿ ಸಂಬಂಧಿತ ವಿಭಾಗಗಳು ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸಿ."
        return "I could not find relevant sections in the document to answer this. Please try rephrasing."

    ctx = "\n\n".join(f"[Source {i}]: {s['text']}" for i, s in enumerate(sources, 1))

    if q_lang == "kn":
        lang_rule = """

CRITICAL LANGUAGE RULE:
The user asked in Kannada. You MUST reply ENTIRELY in Kannada (ಕನ್ನಡ).
Every single word of your answer must be in Kannada script.
Do not use any English words except proper nouns, legal terms, or names that appear in the source.
Write naturally as a fluent Kannada speaker would explain this to another person.
Do NOT transliterate — use proper Kannada Unicode script throughout."""
    else:
        lang_rule = """

Reply in clear, professional English. Use plain language — avoid unnecessary legal jargon."""

    prompt = f"""You are a precise legal document assistant.
Answer the question using ONLY the information in the sources provided below.
Do NOT fabricate or guess. If the answer is not in the sources, say so clearly.
After each factual statement, place a citation marker like [1] or [2] immediately after the relevant sentence.
Format your answer clearly — use numbered lists or paragraphs as appropriate.
Do NOT use ** for bold or * for italic — write clean plain text only.{lang_rule}

{ctx}

Question: {question}

Answer:"""

    return llm_call(prompt, 0.15, 1200)


# ─────────────────────────────────────────────
# SIMPLIFY
# ─────────────────────────────────────────────
def simplify_doc(text: str, lang: str) -> list:
    if not groq_ok():
        return [{"type":"Config Error","original":"","simple":"GROQ_API_KEY not configured."}]
    ln = ("\nDocument is in Kannada. Write all explanations in simple Kannada script."
          if lang == "kn" else "")
    prompt = f"""You are a legal plain-language expert.
Identify 8-12 key provisions from this legal document and explain each simply.{ln}
Write in plain everyday language — no jargon, no ** or * markdown symbols.

Return ONLY a valid JSON array. Each element:
  "type"     - short label (e.g. "Payment Terms")
  "original" - first 100 chars of the relevant clause
  "simple"   - plain explanation in 1-3 sentences

No markdown fences. JSON array only.

Document:
{text[:6000]}"""
    try:
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", llm_call(prompt, 0.2, 2500).strip())
        result = json.loads(raw)
        return result if isinstance(result, list) and result else [
            {"type":"Notice","original":"","simple":"Could not parse. Please try again."}]
    except Exception as ex:
        return [{"type":"Error","original":"","simple":f"Simplification failed: {ex}"}]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div class="sb-brand">
  <div class="sb-icon">⚖️</div>
  <div>
    <div class="sb-name">LexAI</div>
    <div class="sb-sub">Legal Document Assistant</div>
  </div>
</div>""", unsafe_allow_html=True)

        doc_loaded = bool(st.session_state.doc_text)
        if doc_loaded:
            lang  = st.session_state.doc_lang
            flag  = "🇮🇳 Kannada" if lang == "kn" else "🇬🇧 English"
            st.markdown(f"""
<div class="sb-doc-pill">
  <div class="sb-doc-name">📎 {html.escape(st.session_state.doc_name)}</div>
  <div class="sb-doc-lang">{flag}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='sb-nav-label'>Navigation</div>", unsafe_allow_html=True)

        view = st.session_state.view
        if st.button("💬  Chat", key="nav_chat", use_container_width=True,
                     type="primary" if view == "home" else "secondary"):
            st.session_state.view = "home"; st.rerun()

        if st.button("🧠  Simplify Document", key="nav_simp", use_container_width=True,
                     type="primary" if view == "simplify" else "secondary"):
            if doc_loaded:
                st.session_state.view = "simplify"; st.rerun()

        st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)
        if st.button("📄  New Document", key="nav_new", use_container_width=True):
            st.session_state.doc_text     = ""
            st.session_state.doc_name     = ""
            st.session_state.doc_lang     = "en"
            st.session_state.chat_history = []
            st.session_state.simplified   = []
            st.session_state.view         = "home"
            st.rerun()


# ─────────────────────────────────────────────
# UPLOAD PAGE  — shown when no document loaded
# ─────────────────────────────────────────────
def upload_page():
    """Renders ONLY the welcome screen and upload card. No chat widgets at all."""
    st.markdown("""
<div class="welcome-wrap">
  <div class="welcome-icon">⚖️</div>
  <div class="welcome-title">LexAI</div>
  <div class="welcome-desc">
    Upload a legal document and ask questions in natural language.<br>
    Precise answers with references to the exact source text.<br>
    <strong>Supports English &amp; Kannada.</strong>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<div class="upload-label">Upload your document</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">Supported: .txt, .pdf &mdash; English and Kannada</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "", type=["txt", "pdf"],
        label_visibility="collapsed",
        key="file_uploader"
    )

    if uploaded:
        if uploaded.name.lower().endswith(".pdf"):
            text = extract_pdf_text(uploaded.getvalue())
            if not text.strip():
                st.error("Could not extract text from this PDF. It may be image-based or protected.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
        else:
            text = uploaded.getvalue().decode("utf-8", errors="replace")

        if not text.strip():
            st.error("The file appears to be empty.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        lang     = detect_lang(text)
        lang_msg = ("🇮🇳 Kannada document detected." if lang == "kn"
                    else "🇬🇧 English document detected.")
        st.info(f"{lang_msg}  Ask in Kannada or English — answers will match your question language.")

        if st.button("Start Chatting →", type="primary", use_container_width=True, key="start_btn"):
            model = load_model()
            ingest_doc(text, model, uploaded.name)
            st.session_state.doc_text     = text
            st.session_state.doc_name     = uploaded.name
            st.session_state.doc_lang     = lang
            st.session_state.chat_history = []
            st.session_state.simplified   = []
            st.session_state.input_key   += 1
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHAT PAGE  — shown when document is loaded
# ─────────────────────────────────────────────
def chat_page(endee_ok: bool):
    """Renders ONLY the chat interface. No upload widgets at all."""
    doc_text = st.session_state.doc_text
    doc_name = st.session_state.doc_name
    doc_lang = st.session_state.doc_lang

    # Header
    lang_label = "Kannada" if doc_lang == "kn" else "English"
    st.markdown(f"""
<div class="main-header">
  <div class="app-name"><span class="app-name-icon">⚖️</span> LexAI</div>
  <div class="doc-tag">
    <span class="doc-tag-name">📎 {html.escape(doc_name)}</span>
    <span class="doc-tag-sep">·</span>
    <span class="doc-tag-lang">{lang_label}</span>
  </div>
</div>""", unsafe_allow_html=True)

    # Empty state
    if not st.session_state.chat_history:
        ask_label = "ಪ್ರಶ್ನೆ ಕೇಳಿ" if doc_lang == "kn" else "Ask a question"
        st.markdown(f"""
<div style="text-align:center;padding:32px 0 16px;">
  <div style="font-size:0.88rem;color:var(--text2)">
    {ask_label} about <strong style="color:var(--text)">{html.escape(doc_name)}</strong>
  </div>
</div>""", unsafe_allow_html=True)

    # Conversation history
    history = st.session_state.chat_history
    for idx, msg in enumerate(history):
        role    = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        delay   = min(idx * 0.025, 0.2)

        if role == "user":
            st.markdown(f"""
<div class="msg-group" style="animation-delay:{delay}s">
  <div class="msg-row user-row">
    <div class="bubble user">{html.escape(content)}</div>
    <div class="av av-user">You</div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            rendered = render_answer(content, sources)
            st.markdown(f"""
<div class="msg-group" style="animation-delay:{delay}s">
  <div class="msg-row">
    <div class="av av-lex">Lex</div>
    <div class="bubble ai">{rendered}</div>
  </div>
</div>""", unsafe_allow_html=True)

            if sources:
                prev_q = history[idx-1]["content"] if idx > 0 else ""
                src_html = '<div class="sources-panel">'
                for si, src in enumerate(sources, 1):
                    snip = html.escape(src.get("text", "")[:260])
                    for w in prev_q.split():
                        if len(w) > 3:
                            snip = snip.replace(html.escape(w),
                                f"<span class='hl'>{html.escape(w)}</span>")
                    src_html += (f'<div class="src-item" id="src-{si}">'
                                 f'<div class="src-num">[{si}]</div>'
                                 f'<div class="src-snippet">{snip}</div></div>')
                src_html += "</div>"
                st.markdown(src_html, unsafe_allow_html=True)

        if role == "assistant" and idx < len(history) - 1:
            st.markdown("<hr class='msg-divider'>", unsafe_allow_html=True)

    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

    # Input bar
    ph = "ಪ್ರಶ್ನೆ ಕೇಳಿ…" if doc_lang == "kn" else "Ask anything about your document…"
    col_i, col_b = st.columns([11, 1])
    with col_i:
        question = st.text_input(
            "", placeholder=ph, label_visibility="collapsed",
            key=f"qi_{st.session_state.input_key}"
        )
    with col_b:
        send = st.button("↑", type="primary", use_container_width=True)

    if send and question.strip():
        q_lang  = detect_question_lang(question.strip())
        model   = load_model()
        sources = do_search(question.strip(), model) if endee_ok else []
        if not sources:
            fb      = chunk_text(doc_text, 600)[:4]
            sources = [{"doc": doc_name, "text": c, "score": None} for c in fb]
        answer = generate_answer(question.strip(), sources, q_lang)
        st.session_state.chat_history.append({"role": "user",      "content": question.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
        st.session_state.input_key += 1
        st.rerun()


# ─────────────────────────────────────────────
# HOME VIEW  — routes to upload_page or chat_page
# ─────────────────────────────────────────────
def home_view(endee_ok: bool):
    if st.session_state.doc_text:
        chat_page(endee_ok)
    else:
        upload_page()


# ─────────────────────────────────────────────
# SIMPLIFY VIEW
# ─────────────────────────────────────────────
def simplify_view():
    text = st.session_state.doc_text
    lang = st.session_state.doc_lang

    if not text:
        st.info("Upload a document first."); return

    st.markdown(f"""
<div class="sec-header">
  <div class="sec-title">🧠 Plain-Language Summary</div>
  <div class="doc-tag">
    <span class="doc-tag-name">📎 {html.escape(st.session_state.doc_name)}</span>
  </div>
</div>""", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.82rem;color:var(--text2);margin-bottom:18px'>"
                "Key provisions explained in simple, everyday language.</div>",
                unsafe_allow_html=True)

    if not st.session_state.simplified:
        st.session_state.simplified = simplify_doc(text, lang)
        st.rerun()

    for s in st.session_state.simplified:
        ct = html.escape(str(s.get("type", "")))
        og = html.escape(str(s.get("original", ""))[:130])
        sp = html.escape(str(s.get("simple", "")))
        if not sp: continue
        st.markdown(f"""
<div class="simp-card">
  {f'<div class="simp-type">{ct}</div>' if ct else ''}
  {f'<div class="simp-orig">"{og}…"</div>' if og else ''}
  <div class="simp-plain">{sp}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("Regenerate", use_container_width=True):
        st.session_state.simplified = []
        st.rerun()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    inject_theme()
    init_state()
    endee_ok = is_endee_ok()
    render_sidebar()

    if st.session_state.view == "simplify":
        simplify_view()
    else:
        home_view(endee_ok)


if __name__ == "__main__":
    main()