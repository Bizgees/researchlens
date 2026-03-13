import os
import io
import json
import zipfile
import requests
import streamlit as st
from docx import Document
import google.generativeai as genai
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="A4G-ResearchLens · World Article Map",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #161a23;
    --border:  #252b38;
    --accent:  #e8b86d;
    --accent2: #6d9ee8;
    --text:    #e8e6e0;
    --muted:   #7a8099;
    --success: #6de8a8;
}

html, body, .stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }

[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--accent); font-family: 'Playfair Display', serif; }

h1, h2, h3 { font-family: 'Playfair Display', serif; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,184,109,0.15) !important;
}

.stButton > button {
    background: var(--accent);
    color: #0d0f14;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.5rem 1.4rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #f0c97a;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(232,184,109,0.3);
}

.summary-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    line-height: 1.75;
}
.summary-card.enhanced { border-left-color: var(--accent2); }
.summary-card.answer   { border-left-color: var(--success); }

.country-badge {
    display: inline-block;
    background: rgba(232,184,109,0.12);
    border: 1px solid rgba(232,184,109,0.3);
    color: var(--accent);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.2rem;
}
.article-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    cursor: pointer;
    transition: border-color 0.2s;
}
.article-card:hover { border-color: var(--accent); }
.article-card.selected { border-color: var(--accent); border-left: 3px solid var(--accent); }

.section-label {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.chat-bubble {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
}
.chat-bubble.user { border-left: 3px solid var(--accent2); }
.chat-bubble.ai   { border-left: 3px solid var(--success); }
.chat-label { font-size: 0.72rem; color: var(--muted); margin-bottom: 0.3rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; }

hr { border-color: var(--border); }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Playfair Display', serif; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MAX_CHARS = 3000


# ─────────────────────────────────────────────
# AI HELPERS
# ─────────────────────────────────────────────
def build_llm(gemini_key: str, serper_key: str):
    """Configure Gemini and return a callable model."""
    os.environ["GOOGLE_API_KEY"] = gemini_key
    os.environ["SERPER_API_KEY"] = serper_key
    genai.configure(api_key=gemini_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-04-17",
        generation_config=genai.GenerationConfig(temperature=0.2),
    )

def _ask(llm, prompt: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    response = llm.generate_content(prompt)
    return response.text.strip()

def extract_country(llm, article_text: str, file_name: str) -> str:
    """Use AI to extract the primary country an article is about."""
    prompt = f"""Read the following article excerpt and identify the PRIMARY country it is about.

Rules:
- Return ONLY the country name as it appears in standard ISO naming (e.g. "United States", "United Kingdom", "South Africa").
- If the article is about multiple countries, return the most prominently featured one.
- If no specific country is identifiable, return "Unknown".
- Do NOT return a sentence, explanation, or punctuation — just the country name.

File name hint: {file_name}

Article excerpt:
{article_text[:1500]}"""
    result = _ask(llm, prompt)
    for prefix in ["The country is", "Country:", "Answer:", "Primary country:"]:
        if result.lower().startswith(prefix.lower()):
            result = result[len(prefix):].strip()
    return result.strip(".,\n\"' ")

def run_summary(llm, article_text: str) -> str:
    prompt = f"""Summarize the following research article in 5-6 clear sentences.
Focus only on the main idea and key findings from the article text.
Do NOT add outside information.

Article:
{article_text}"""
    return _ask(llm, prompt)

def run_enhanced_summary(llm, article_text: str, web_results: str, topic: str) -> str:
    prompt = f"""You have two sources about "{topic}":

1. LOCAL ARTICLE:
{article_text}

2. WEB SEARCH RESULTS (new external information):
{web_results}

Write an enhanced summary (6-8 sentences) that:
- Starts with the local article's main findings
- Then adds NEW context and insights from the web results
- Clearly distinguishes between what the article says vs what the web adds
- Does NOT simply repeat the local summary"""
    return _ask(llm, prompt)

def run_qa(llm, article_text: str, question: str, web_results: str = "") -> str:
    context = f"LOCAL ARTICLE:\n{article_text}"
    if web_results:
        context += f"\n\nWEB SEARCH CONTEXT:\n{web_results}"
    prompt = f"""Answer the user's question using the context below.
Be specific and direct. If the answer is not in the context, say so clearly.

CONTEXT:
{context}

USER QUESTION:
{question}"""
    return _ask(llm, prompt)


# ─────────────────────────────────────────────
# ARTICLE LOADING & CACHE
# ─────────────────────────────────────────────
BUNDLED_FOLDER = "articles"        # folder in GitHub repo alongside app file
CACHE_FILE     = "article_cache.json"  # persists country detection results

def load_cache() -> dict:
    """Load the article cache (file_name -> country) from disk."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cache(cache: dict):
    """Save the article cache to disk."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def parse_docx_bytes(file_bytes: bytes, file_name: str) -> dict | None:
    """Parse a .docx file from bytes into an article dict."""
    try:
        doc = Document(io.BytesIO(file_bytes))
        text = " ".join(p.text for p in doc.paragraphs)
        wc = len(text.split())
        if wc >= 100:
            return {"file_name": file_name, "text": text, "word_count": wc, "country": None}
    except Exception:
        pass
    return None

def load_from_folder(folder_path: str) -> list:
    """Load .docx files from a local folder path."""
    articles = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".docx"):
                path = os.path.join(root, file)
                try:
                    with open(path, "rb") as f:
                        art = parse_docx_bytes(f.read(), file)
                    if art:
                        art["source"] = "bundled"
                        articles.append(art)
                except Exception:
                    pass
    return articles

def load_from_zip(zip_file) -> list:
    """Extract and load .docx files from an uploaded ZIP."""
    articles = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
            for name in zf.namelist():
                fname = os.path.basename(name)
                if fname.lower().endswith(".docx") and not fname.startswith("__"):
                    with zf.open(name) as f:
                        art = parse_docx_bytes(f.read(), fname)
                    if art:
                        art["source"] = "uploaded"
                        articles.append(art)
    except Exception as e:
        st.error(f"ZIP error: {e}")
    return articles


# ─────────────────────────────────────────────
# MAP BUILDER
# ─────────────────────────────────────────────
def build_world_map(articles_with_countries: list):
    """Build a Plotly choropleth world map highlighting countries in the articles."""
    country_counts = {}
    country_articles = {}
    for art in articles_with_countries:
        c = art.get("country", "Unknown")
        if c and c != "Unknown":
            country_counts[c] = country_counts.get(c, 0) + 1
            country_articles.setdefault(c, []).append(art["file_name"])

    countries  = list(country_counts.keys())
    counts     = [country_counts[c] for c in countries]
    hover_text = [
        f"<b>{c}</b><br>{country_counts[c]} article(s)<br><br>" +
        "<br>".join(f"• {a}" for a in country_articles[c])
        for c in countries
    ]

    fig = go.Figure(go.Choropleth(
        locations=countries,
        locationmode="country names",
        z=counts,
        colorscale=[[0, "#1e2535"], [0.5, "#c4933a"], [1.0, "#e8b86d"]],
        zmin=0,
        zmax=max(counts) if counts else 1,
        showscale=False,
        hovertemplate="%{text}<extra></extra>",
        text=hover_text,
        marker_line_color="#252b38",
        marker_line_width=0.5,
    ))

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#252b38",
            showland=True,
            landcolor="#161a23",
            showocean=True,
            oceancolor="#0d0f14",
            showcountries=True,
            countrycolor="#252b38",
            projection_type="natural earth",
            bgcolor="#0d0f14",
        ),
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#0d0f14",
        margin=dict(l=0, r=0, t=0, b=0),
        height=460,
        hoverlabel=dict(
            bgcolor="#161a23",
            bordercolor="#e8b86d",
            font=dict(color="#e8e6e0", family="DM Sans"),
        ),
    )
    return fig, country_articles


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "llm": None,
    "articles": [],           # list of dicts with country assigned
    "country_articles": {},   # {country: [file_names]}
    "selected_country": None,
    "selected_article": None,
    "summary": None,
    "enhanced_summary": None,
    "web_results": "",
    "web_search_done": False,
    "chat_history": [],
    "processing_countries": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# HELPERS — process articles with AI country detection + caching
# ─────────────────────────────────────────────
def process_articles(raw_articles: list, llm, progress_widget=None) -> tuple:
    """
    Run AI country extraction only for articles not already in cache.
    Saves results back to cache so they are never re-processed.
    Returns (articles, country_articles).
    """
    cache = load_cache()
    new_extractions = 0

    # Separate articles needing AI vs already cached
    need_ai = [a for a in raw_articles if a["file_name"] not in cache]
    cached  = [a for a in raw_articles if a["file_name"] in cache]

    # Apply cached countries immediately
    for art in cached:
        art["country"] = cache[art["file_name"]]

    # Run AI only on uncached articles
    total = len(need_ai)
    for i, art in enumerate(need_ai):
        if progress_widget and total > 0:
            progress_widget.progress(i / total, text=f"🤖 Detecting country: {art['file_name']}")
        country = extract_country(llm, art["text"], art["file_name"])
        art["country"] = country
        cache[art["file_name"]] = country
        new_extractions += 1

    # Save updated cache
    if new_extractions > 0:
        save_cache(cache)

    if progress_widget:
        progress_widget.progress(1.0, text=f"✅ Done! ({new_extractions} new, {len(cached)} from cache)")

    all_articles = cached + need_ai
    ca = {}
    for art in all_articles:
        c = art.get("country", "Unknown")
        ca.setdefault(c, []).append(art)
    return all_articles, ca


def rebuild_country_index(articles: list) -> dict:
    """Rebuild the country → articles index from a list of articles."""
    ca = {}
    for art in articles:
        c = art.get("country", "Unknown")
        ca.setdefault(c, []).append(art)
    return ca


def auto_load_bundled(llm) -> tuple:
    """Load bundled articles from the articles/ folder, using cache for country detection."""
    raw = load_from_folder(BUNDLED_FOLDER)
    if not raw:
        return [], {}
    arts, ca = process_articles(raw, llm)
    return arts, ca


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 A4G-ResearchLens")
    st.markdown("<p style='color:#7a8099;font-size:0.85rem;margin-top:-0.5rem'>World Article Explorer</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── API Keys ──
    st.markdown("### ⚙️ API Keys")
    try:
        default_gemini = st.secrets.get("GOOGLE_API_KEY", "")
        default_serper = st.secrets.get("SERPER_API_KEY", "")
    except Exception:
        default_gemini = ""
        default_serper = ""
    gemini_key = st.text_input("Google Gemini API Key", value=default_gemini, type="password", placeholder="AIza...")
    serper_key  = st.text_input("Serper API Key", value=default_serper, type="password", placeholder="Your Serper key")
    st.markdown("<p style='color:#7a8099;font-size:0.75rem'>Get a free Gemini key at <a href='https://aistudio.google.com' target='_blank' style='color:#e8b86d'>aistudio.google.com</a></p>", unsafe_allow_html=True)

    # ── Auto-load bundled articles on startup ──
    keys_ready = bool(gemini_key and serper_key)
    bundled_exists = os.path.exists(BUNDLED_FOLDER)

    if keys_ready and bundled_exists and not st.session_state.articles:
        st.session_state.llm = build_llm(gemini_key, serper_key)
        with st.spinner("Loading articles…"):
            prog = st.progress(0, text="Checking article cache…")
            arts, ca = auto_load_bundled(st.session_state.llm)
            prog.progress(1.0, text="Done!")
        st.session_state.articles = arts
        st.session_state.country_articles = ca

    st.markdown("---")

    # ── Bundled articles status ──
    st.markdown("### 📚 Bundled Articles")
    if bundled_exists:
        bundled_count = sum(1 for _, _, fs in os.walk(BUNDLED_FOLDER) for f in fs if f.lower().endswith(".docx"))
        cached_count  = len(load_cache())
        st.markdown(
            f"<p style='color:#6de8a8;font-size:0.84rem'>✅ {bundled_count} article(s) in repo</p>"
            f"<p style='color:#7a8099;font-size:0.78rem'>⚡ {cached_count} country detection(s) cached</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<p style='color:#e87a6d;font-size:0.82rem'>⚠️ No <code>articles/</code> folder found.</p>", unsafe_allow_html=True)

    reload_btn = st.button("🔄 Reload Bundled Articles", use_container_width=True, disabled=not (keys_ready and bundled_exists))
    if reload_btn:
        if not keys_ready:
            st.error("Enter both API keys first.")
        else:
            st.session_state.llm = build_llm(gemini_key, serper_key)
            for k in ["articles","country_articles","selected_country",
                      "selected_article","summary","enhanced_summary",
                      "web_results","web_search_done","chat_history"]:
                st.session_state[k] = defaults[k]
            prog = st.progress(0, text="Loading…")
            arts, ca = auto_load_bundled(st.session_state.llm)
            prog.progress(1.0, text="Done!")
            st.session_state.articles = arts
            st.session_state.country_articles = ca
            st.success(f"✅ {len(arts)} articles loaded!")

    st.markdown("---")

    # ── ZIP upload for new articles ──
    st.markdown("### ➕ Add New Articles")
    st.markdown(
        "<p style='color:#7a8099;font-size:0.82rem'>"
        "Zip your articles folder and upload it here.<br>"
        "Only NEW articles (not already in the library) will be processed.</p>",
        unsafe_allow_html=True
    )
    zip_file = st.file_uploader("Upload ZIP", type=["zip"], label_visibility="collapsed")
    upload_btn = st.button("➕ Add Articles from ZIP", use_container_width=True, disabled=not zip_file)

    if upload_btn and zip_file:
        if not keys_ready:
            st.error("Enter both API keys first.")
        else:
            if not st.session_state.llm:
                st.session_state.llm = build_llm(gemini_key, serper_key)
            with st.spinner("Extracting ZIP…"):
                new_arts = load_from_zip(zip_file)

            if not new_arts:
                st.error("No valid .docx files found in the ZIP.")
            else:
                existing_names = {a["file_name"] for a in st.session_state.articles}
                truly_new = [a for a in new_arts if a["file_name"] not in existing_names]
                if not truly_new:
                    st.warning("All articles in the ZIP are already in the library.")
                else:
                    prog = st.progress(0, text="Processing new articles…")
                    truly_new, _ = process_articles(truly_new, st.session_state.llm, prog)
                    st.session_state.articles += truly_new
                    st.session_state.country_articles = rebuild_country_index(st.session_state.articles)
                    st.success(f"✅ {len(truly_new)} new article(s) added!")
                    st.info("💡 To make these permanent, add the .docx files to your articles/ folder on GitHub.")

    # ── Country index ──
    if st.session_state.country_articles:
        st.markdown("---")
        st.markdown("### 🗺 Countries Found")
        for country, arts in sorted(st.session_state.country_articles.items()):
            if country != "Unknown":
                st.markdown(
                    f"<div style='font-size:0.82rem;padding:0.25rem 0;color:#7a8099'>"
                    f"📍 <span style='color:#e8b86d'>{country}</span> · {len(arts)} article(s)</div>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("<h1 style='margin-bottom:0.1rem'>A4G-ResearchLens</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#7a8099;margin-top:0;margin-bottom:1.5rem'>Explore research articles by country · Summarize · Ask follow-up questions</p>", unsafe_allow_html=True)

if not st.session_state.articles:
    st.info("👈  Enter your API keys in the sidebar — articles will load automatically.")
    st.stop()


# ─────────────────────────────────────────────
# WORLD MAP
# ─────────────────────────────────────────────
st.markdown("### 🗺 Articles by Country")
st.markdown("<p style='color:#7a8099;font-size:0.88rem;margin-top:-0.5rem'>Hover over a highlighted country to see articles. Then select a country below to explore.</p>", unsafe_allow_html=True)

fig, country_articles_map = build_world_map(st.session_state.articles)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Country selector below map ──
known_countries = sorted([c for c in st.session_state.country_articles.keys() if c != "Unknown"])
unknown_articles = st.session_state.country_articles.get("Unknown", [])

if known_countries:
    col_select, col_gap = st.columns([3, 2])
    with col_select:
        selected_country = st.selectbox(
            "Select a country to explore its articles",
            options=["— choose a country —"] + known_countries,
            label_visibility="collapsed",
        )
    if selected_country != "— choose a country —" and selected_country != st.session_state.selected_country:
        st.session_state.selected_country = selected_country
        st.session_state.selected_article = None
        st.session_state.summary = None
        st.session_state.enhanced_summary = None
        st.session_state.web_results = ""
        st.session_state.web_search_done = False
        st.session_state.chat_history = []
        st.rerun()

if unknown_articles:
    with st.expander(f"⚠️ {len(unknown_articles)} article(s) with undetected country"):
        for a in unknown_articles:
            st.markdown(f"- {a['file_name']}")


# ─────────────────────────────────────────────
# ARTICLE LIST FOR SELECTED COUNTRY
# ─────────────────────────────────────────────
if st.session_state.selected_country:
    country = st.session_state.selected_country
    arts = st.session_state.country_articles.get(country, [])

    st.markdown("---")
    st.markdown(f"### 📄 Articles · {country}")
    st.markdown(f"<p style='color:#7a8099;font-size:0.88rem;margin-top:-0.4rem'>{len(arts)} article(s) found — click one to summarize</p>", unsafe_allow_html=True)

    for art in arts:
        is_selected = (
            st.session_state.selected_article is not None and
            st.session_state.selected_article["file_name"] == art["file_name"]
        )
        card_class = "article-card selected" if is_selected else "article-card"
        st.markdown(
            f"<div class='{card_class}'>"
            f"<span style='color:#e8b86d;font-weight:500'>📄 {art['file_name']}</span>"
            f"<span style='color:#7a8099;font-size:0.8rem;margin-left:0.8rem'>{art['word_count']:,} words</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button(f"Select · {art['file_name']}", key=f"sel_{art['file_name']}"):
            st.session_state.selected_article = art
            st.session_state.summary = None
            st.session_state.enhanced_summary = None
            st.session_state.web_results = ""
            st.session_state.web_search_done = False
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
if st.session_state.selected_article:
    art = st.session_state.selected_article
    article_text = art["text"][:MAX_CHARS]

    st.markdown("---")
    st.markdown(f"### ✨ Summary · {art['file_name']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Country", art.get("country", "Unknown"))
    col2.metric("Word Count", f"{art['word_count']:,}")
    col3.metric("Chars sent to AI", f"{len(article_text):,}")

    if not st.session_state.summary:
        if st.button("✨ Generate Summary"):
            with st.spinner("Summarising article…"):
                st.session_state.summary = run_summary(st.session_state.llm, article_text)
            st.rerun()

    if st.session_state.summary:
        st.markdown("<div class='section-label'>Article Summary</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='summary-card'>{st.session_state.summary}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# WEB ENRICHMENT
# ─────────────────────────────────────────────
if st.session_state.summary:
    art = st.session_state.selected_article
    article_text = art["text"][:MAX_CHARS]

    st.markdown("---")
    st.markdown("### 🌐 Enhance with Web Search")

    if not st.session_state.web_search_done:
        st.markdown("<p style='color:#7a8099;font-size:0.88rem'>Search the internet for additional context and generate an enriched summary.</p>", unsafe_allow_html=True)
        if st.button("🌐 Search Web & Enhance Summary"):
            search_term = art["file_name"].rsplit(".", 1)[0]
            with st.spinner("Searching the web…"):
                try:
                    serper_api_key = os.environ.get("SERPER_API_KEY", "")
                    resp = requests.post(
                        "https://google.serper.dev/search",
                        headers={"X-API-KEY": serper_api_key, "Content-Type": "application/json"},
                        json={"q": search_term, "num": 5},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    snippets = []
                    for item in data.get("organic", []):
                        title   = item.get("title", "")
                        snippet = item.get("snippet", "")
                        link    = item.get("link", "")
                        if snippet:
                            snippets.append(f"- {title}: {snippet} ({link})")
                    web_results = "\n".join(snippets) if snippets else ""
                    st.session_state.web_results = web_results
                except Exception as e:
                    st.error(f"Web search failed: {e}")
                    web_results = ""

            if web_results:
                with st.spinner("Generating enhanced summary…"):
                    st.session_state.enhanced_summary = run_enhanced_summary(
                        st.session_state.llm,
                        article_text,
                        web_results,
                        search_term,
                    )
                    st.session_state.web_search_done = True
                    st.rerun()
    else:
        st.success("✅ Web-enhanced summary ready.")

    if st.session_state.enhanced_summary:
        st.markdown("<div class='section-label'>Enhanced Summary</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='summary-card enhanced'>{st.session_state.enhanced_summary}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Q&A
# ─────────────────────────────────────────────
if st.session_state.summary:
    art = st.session_state.selected_article
    article_text = art["text"][:MAX_CHARS]

    st.markdown("---")
    st.markdown("### 💬 Ask Follow-up Questions")
    st.markdown("<p style='color:#7a8099;font-size:0.88rem;margin-top:-0.4rem'>Ask anything about this article. Web context is included if you ran the web search.</p>", unsafe_allow_html=True)

    # Chat history
    for turn in st.session_state.chat_history:
        st.markdown(f"<div class='chat-bubble user'><div class='chat-label'>You</div>{turn['q']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble ai'><div class='chat-label'>A4G-ResearchLens</div>{turn['a']}</div>", unsafe_allow_html=True)

    qa_col, btn_col = st.columns([5, 1])
    with qa_col:
        user_q = st.text_input(
            "question",
            placeholder="e.g. What are the key findings? Who is most affected? What solutions are proposed?",
            label_visibility="collapsed",
            key="qa_input",
        )
    with btn_col:
        ask_btn = st.button("Ask", use_container_width=True)

    if ask_btn and user_q.strip():
        with st.spinner("Thinking…"):
            answer = run_qa(
                st.session_state.llm,
                article_text,
                user_q,
                st.session_state.web_results if st.session_state.web_search_done else "",
            )
            st.session_state.chat_history.append({"q": user_q, "a": answer})
            st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
