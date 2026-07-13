import os
import re
import base64
import streamlit as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv
from fpdf import FPDF

# Load environment variables from .env file
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Pebble - Small questions. Trusted answers.",
    page_icon="pebble_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Listen for the 'New Chat' hyperlink trigger
if "reset" in st.query_params:
    st.session_state.chat_history = []
    st.query_params.clear()

import streamlit.components.v1 as components

components.html(
    """
    <script>
        const parentDoc = window.parent.document;
        
        // 1. CSS Scroll Lock: Physically prevent the browser from scrolling
        const style = parentDoc.createElement('style');
        style.innerHTML = `
            .no-scroll-load {
                overflow: hidden !important;
            }
        `;
        parentDoc.head.appendChild(style);

        // Apply the lock to the main container immediately
        const viewContainer = parentDoc.querySelector('[data-testid="stAppViewContainer"]') || parentDoc.querySelector('.main');
        if (viewContainer) {
            viewContainer.classList.add('no-scroll-load');
            viewContainer.scrollTop = 0;
        }
        window.parent.scrollTo(0, 0);

        // 2. DOM Observer: Hunt down the specific chat input and neutralize it
        const observer = new MutationObserver((mutations, obs) => {
            // Target Streamlit's specific chat text area
            const chatInput = parentDoc.querySelector('[data-testid="stChatInputTextArea"]');
            if (chatInput) {
                // The exact millisecond it renders, remove its focus
                chatInput.blur(); 
                
                // Force scroll to top one last time
                if (viewContainer) viewContainer.scrollTop = 0; 
                
                // Mission accomplished, stop watching the DOM
                obs.disconnect(); 
            }
        });
        
        // Start watching the entire Streamlit app for changes
        observer.observe(parentDoc.body, { childList: true, subtree: true });

        // 3. Cleanup: Release the scroll lock after 2 seconds
        setTimeout(function() {
            if (viewContainer) viewContainer.classList.remove('no-scroll-load');
            style.remove();
            observer.disconnect(); // Failsafe
        }, 2000);
    </script>
    """,
    height=0,
    width=0,
)

# Custom Premium Styling (Neuropelvic Surgery Corporate Style with Custom Overrides)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap');

    @font-face {
        font-family: 'Aspekta';
        src: url('https://neuropelvicsurgery.com/wp-content/themes/Pelvic%20Surgery/static/fonts/Aspekta-500.woff2') format('woff2');
        font-weight: 500;
        font-style: normal;
        font-display: swap;
    }
    
    /* Global Styles */
    .stApp {
        font-family: 'Aspekta', sans-serif;
        background: #FFFFFF; /* Ultra-Clean White Background */
        color: #121212; /* Dark Charcoal Text */
    }
    
    /* Remove excessive whitespace at the top of the main container */
    .block-container {
        padding-top: 1rem !important;
    }
    
    /* 1. Typography Override: DM Sans for headings and subheadings, Plus Jakarta Sans for description paragraphs */
    h1, h2, h3, h4, h5, h6, .hero-title, .hero-subtitle {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700;
        color: #121212 !important;
        letter-spacing: -0.02em;
        font-style: normal !important;
    }
    
    .hero-description, p {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 500;
        color: #121212 !important;
        letter-spacing: -0.02em;
        font-style: italic !important;
    }
    
    /* 2. Top Bar Removal */
    [data-testid="stHeader"] {
        display: none !important;
    }
    
    /* 3. Sidebar Removal */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Header Nav Bar */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: none;
        margin-bottom: 0.5rem;
    }
    .nav-logo {
        font-size: 1.25rem;
        font-weight: 700;
        color: #121212;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    .nav-link, .nav-link:visited, .nav-link:active {
        font-size: 0.95rem;
        font-weight: 400;
        color: #000000 !important;
        text-decoration: none !important;
        transition: color 0.2s, text-decoration 0.2s;
    }
    .nav-link:hover {
        color: #007BFF !important;
        text-decoration: underline !important;
    }
    
    /* MOBILE OPTIMIZATION ENGINE */
    @media (max-width: 768px) {
        .nav-container {
            flex-direction: column;
            gap: 16px;
            align-items: center;
            text-align: center;
        }
        .nav-links {
            flex-wrap: wrap;
            justify-content: center;
            gap: 16px;
            font-size: 12px;
        }
        .hero-title {
            font-size: 2rem !important;
        }
        .hero-subtitle, .hero-description {
            font-size: 1rem !important;
        }
    }
    
    /* Hero Container */
    .hero-container {
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #F0F0F0;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 1rem;
        color: #121212;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        color: #666666;
        line-height: 1.6;
        max-width: 850px;
    }
    
    /* Info/Status Cards (Flat Corporate) */
    .status-card {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.01);
        margin-bottom: 1.5rem;
    }
    
    /* Chat Bubble Styling (Flat, Clean, Modern) */
    .chat-bubble {
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        max-width: 90%;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    .chat-user {
        background: #F5F5F7; /* Clean Light Grey */
        color: #121212;
        margin-left: auto;
        border-radius: 12px 12px 0 12px;
    }
    
    .chat-bot {
        background: #FFFFFF;
        color: #121212;
        margin-right: auto;
        border-radius: 12px 12px 12px 0;
        border: 1px solid #E5E5E5;
    }
    
    /* Match Card in Local Mode (Clean Flat Card) */
    .match-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #E5E5E5;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .match-card:hover {
        border-color: #FF5900;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    }
    .match-title {
        font-family: 'Aspekta', sans-serif;
        font-weight: 700;
        color: #121212 !important;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .match-url {
        font-size: 0.85rem;
        color: #FF5900;
        text-decoration: none;
        margin-bottom: 1rem;
        display: inline-block;
        font-weight: 500;
    }
    .match-text {
        font-size: 0.95rem;
        color: #444444;
        line-height: 1.6;
        background: #F9F9FB;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #EEEEEE;
    }
    
    /* Custom Button (Corporate Flat Black, Orange Hover) */
    .stButton>button {
        background-color: #121212 !important; /* Flat Black */
        color: #FFFFFF !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Aspekta', sans-serif;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: background-color 0.2s, transform 0.1s !important;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #FF5900 !important; /* Signature Orange */
    }
    .stButton>button:active {
        transform: scale(0.98);
    }
    
    /* 4. Bottom Bar Trim: Target the bottom chat input container, reduce height, strip dark padding, trim margins */
    div[data-testid="stBottom"],
    .stBottom,
    div[data-testid="stBottom"] > div,
    [data-testid="stBottom"] [data-testid="stChatInputContainer"],
    [data-testid="stBottom"] .stChatInputContainer {
        background-color: #EBF4F8 !important;
        background: #EBF4F8 !important;
    }

    [data-testid="stChatInputContainer"], .stChatInputContainer {
        padding: 0.5rem 0 0 0 !important;
        margin: 0 !important;
        background-color: #EBF4F8 !important;
        background: #EBF4F8 !important;
        border: none !important;
        box-shadow: none !important;
        height: auto !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* Bottom Soft Blue Banner containing reference line */
    div[data-testid="stBottom"] [data-testid="stChatInputContainer"]::after,
    div[data-testid="stBottom"] .stChatInputContainer::after,
    [data-testid="stChatInputContainer"]::after,
    .stChatInputContainer::after {
        content: "Responses reference content from HSE MyChild (https://www2.hse.ie/my-child/)." !important;
        display: block !important;
        text-align: center !important;
        background-color: #EBF4F8 !important;
        background: #EBF4F8 !important;
        color: #121212 !important;
        -webkit-text-fill-color: #121212 !important; /* Webkit text override */
        opacity: 1 !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.85rem !important;
        font-family: 'Aspekta', sans-serif !important;
        font-style: italic !important;
        width: 100vw !important;
        box-sizing: border-box !important;
        margin-top: 0.5rem !important;
    }
    .stChatInput {
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.5rem 0 !important;
    }
    [data-testid="stChatInput"] > div, .stChatInput > div {
        margin: 0 !important;
        padding: 4px !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E5E5E5 !important;
        border-radius: 50px !important;
    }
    
    /* Fix chat input internal textarea & immediate wrapper backgrounds to white and text to black */
    .stChatInput textarea,
    [data-testid="stChatInput"] textarea,
    [data-baseweb="textarea"],
    [data-baseweb="textarea"] > div,
    [data-baseweb="base-input"],
    [data-baseweb="base-input"] > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important; /* Forces Safari/WebKit color override */
        caret-color: #000000 !important; /* Forces blinking cursor to be black */
    }
    
    /* Ensure the placeholder text is visible (gray) against the white background */
    .stChatInput textarea::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #757575 !important;
        -webkit-text-fill-color: #757575 !important;
        opacity: 1 !important;
    }
    
    /* 5. Premium Button Styling: submit button inside chat input as a blue pill with dark circle send icon inside */
    button[data-testid="stChatInputButton"], .stChatInput button {
        border-radius: 50px !important;
        background: linear-gradient(135deg, #0055ff 0%, #00aaff 100%) !important;
        width: 75px !important;
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0, 85, 255, 0.25) !important;
        position: relative !important;
    }
    button[data-testid="stChatInputButton"] svg, .stChatInput button svg {
        background-color: #121212 !important;
        border-radius: 50% !important;
        padding: 5px !important;
        width: 26px !important;
        height: 26px !important;
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        margin-left: 24px !important; /* rests inside the right edge of the blue pill */
    }
    
    /* Ensure no borders, padding, or backgrounds for Streamlit image containers */
    .stImage, [data-testid="stImage"], [data-testid="stImage"] img {
        border: none !important;
        background: transparent !important;
        background-color: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    
    [data-testid="stImage"] img {
        -webkit-mask-image: linear-gradient(to right, black 85%, transparent 100%), linear-gradient(to bottom, black 85%, transparent 100%);
        -webkit-mask-composite: source-in;
        mask-image: linear-gradient(to right, black 85%, transparent 100%), linear-gradient(to bottom, black 85%, transparent 100%);
        mask-composite: intersect;
    }
    
    /* Completely hide the image fullscreen/zoom button on hover */
    [data-testid="stImage"] button,
    .stImage button,
    button[title="View fullscreen"],
    [data-testid="StyledFullScreenButton"],
    [data-testid="stImageHoverContainer"] button,
    .element-container button[title="View fullscreen"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
    }
    button[title="View fullscreen"] { display: none !important; }
    [data-testid="StyledFullScreenButton"] { display: none !important; }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_hse_data(file_path):
    if not os.path.exists(file_path):
        return None
        
    pages = []
    current_page = {}
    content_lines = []
    in_content = False
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str == "=== PAGE START ===":
                current_page = {}
                content_lines = []
                in_content = False
            elif line_str == "=== PAGE END ===":
                if current_page:
                    current_page["content"] = "\n".join(content_lines).strip()
                    pages.append(current_page)
                current_page = {}
                content_lines = []
                in_content = False
            elif line_str.startswith("URL:"):
                current_page["url"] = line_str[4:].strip()
            elif line_str.startswith("TITLE:"):
                current_page["title"] = line_str[6:].strip()
            elif line_str.startswith("INDEX:"):
                current_page["index"] = line_str[6:].strip()
            elif line_str == "CONTENT:":
                in_content = True
            elif in_content:
                content_lines.append(line.rstrip("\n"))
                
    return pages

# Chunk pages for precise RAG matching
@st.cache_data
def prepare_chunks(pages):
    chunks = []
    for page in pages:
        # Split content by double newlines or paragraph patterns
        paragraphs = re.split(r'\n\s*\n', page["content"])
        current_chunk = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Simple chunking logic to group paragraphs together to ~600 chars
            current_chunk.append(para)
            current_len += len(para)
            
            if current_len >= 600:
                chunks.append({
                    "url": page["url"],
                    "title": page["title"],
                    "text": "\n\n".join(current_chunk)
                })
                current_chunk = []
                current_len = 0
                
        # Append remaining paragraphs
        if current_chunk:
            chunks.append({
                "url": page["url"],
                "title": page["title"],
                "text": "\n\n".join(current_chunk)
            })
            
    return chunks

# Build TF-IDF Vectorizer
@st.cache_resource
def build_tfidf_index(chunks):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def search_relevant_chunks(query, chunks, vectorizer, tfidf_matrix, top_k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Lower threshold for short queries (like a single word) to ensure we still return matches
    threshold = 0.01 if len(query.strip().split()) <= 2 else 0.05
    
    results = []
    for idx in top_indices:
        if similarities[idx] > threshold:  # Relevance threshold
            results.append((chunks[idx], similarities[idx]))
    return results

# Load and base64-encode the logo and hero images
logo_path = "pebble_logo.png"
logo_base64 = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

flower_path = "hero_flower.png"
flower_base64 = ""
if os.path.exists(flower_path):
    with open(flower_path, "rb") as f:
        flower_base64 = base64.b64encode(f.read()).decode()

# Main Application Layout
export_href = "#"
if "chat_history" in st.session_state and len(st.session_state.chat_history) > 0:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(200, 10, text="Pebble: Maternal Insight Network - Chat Export", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.cell(200, 10, text="", new_x="LMARGIN", new_y="NEXT") # empty line
    for msg in st.session_state.chat_history:
        role = "Pebble" if msg["role"] == "bot" else "You"
        # use multi_cell to wrap text
        pdf.set_font("helvetica", style="B", size=12)
        pdf.cell(200, 8, text=f"{role}:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", size=12)
        # handle special characters by ignoring or replacing
        clean_text = msg['content'].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, text=clean_text)
        pdf.cell(200, 5, text="", new_x="LMARGIN", new_y="NEXT") # spacing
        
    pdf_bytes = pdf.output()
    encoded_export = base64.b64encode(pdf_bytes).decode('utf-8')
    export_href = f"data:application/pdf;base64,{encoded_export}"

st.markdown(f"""
<style>
/* Modal Base */
.pebble-modal {{
    display: none;
    position: fixed;
    z-index: 99999;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.5);
    backdrop-filter: blur(4px);
}}

/* Show modal when targeted */
.pebble-modal:target {{
    display: flex;
    align-items: center;
    justify-content: center;
}}

/* Modal Content */
.pebble-modal-content {{
    background-color: #ffffff;
    padding: 32px;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    max-width: 450px;
    position: relative;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    font-family: 'Aspekta', sans-serif;
    text-align: left;
}}

/* Close Button */
.pebble-modal-close {{
    position: absolute;
    right: 20px;
    top: 15px;
    color: #94a3b8;
    font-size: 24px;
    font-weight: bold;
    text-decoration: none;
    transition: color 0.2s;
}}
.pebble-modal-close:hover {{
    color: #334155;
    text-decoration: none;
}}

.pebble-modal-title {{
    font-size: 1.25rem;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 16px;
    margin-top: 0;
}}
.pebble-modal-body {{
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.6;
    margin: 0;
}}
</style>

<div class="nav-container">
    <div class="nav-logo" style="display: flex; align-items: center; gap: 10px;">
    </div>
    <div class="nav-links">
        <a class="nav-link" href="#how-pebble-works">How Pebble Works</a>
        <a class="nav-link" href="/?reset=true" target="_self">New Chat</a>
        <a class="nav-link" href="{export_href}" download="Pebble_Chat_Export.pdf">Export Chat</a>
        <a class="nav-link" href="https://www2.hse.ie/my-child/" target="_blank">HSE MyChild Official Website</a>
    </div>
</div>

<div id="how-pebble-works" class="pebble-modal">
    <div class="pebble-modal-content">
        <a href="#" class="pebble-modal-close">&times;</a>
        <h3 class="pebble-modal-title">How Pebble Works</h3>
        <p class="pebble-modal-body">
            Pebble does not search the open web. It only reads and synthesizes verified documents provided by the HSE mychild.ie portal.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

hero_html = f"""
<style>
.hero-flex {{
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 2rem;
    width: 100%;
}}
.hero-image-col {{
    flex: 0 0 auto;
    width: 25%;
    display: flex;
    justify-content: flex-start;
    align-items: center;
}}
.hero-text-col {{
    flex: 1;
    min-width: 0; /* Prevents flex children from overflowing */
}}

.animated-heading-container h1 {{
    display: flex;
    align-items: center;
    flex-wrap: wrap; /* Fix overflow on mobile */
    gap: 12px;
    margin-bottom: 0.5rem;
}}
.tagline-divider {{
    font-weight: 300;
    font-size: 1em;
    color: #cbd5e1;
    transform: translateY(3px);
}}
.typewriter-text {{
    display: inline-block;
    overflow: hidden;
    white-space: nowrap;
    font-weight: 400; 
    font-size: 0.45em;
    color: #64748b;
    border-right: 2px solid #64748b; 
    transform: translateY(3px);
    line-height: 1.2;
    animation: type-and-pause 5.5s infinite, blink 0.8s step-end infinite;
}}

@keyframes type-and-pause {{
    0% {{ max-width: 0; animation-timing-function: steps(25, end); }}
    45.45% {{ max-width: 26ch; animation-timing-function: step-end; }}
    99% {{ max-width: 26ch; }}
    100% {{ max-width: 0; }}
}}

@keyframes blink {{
    from, to {{ border-color: transparent; }}
    50% {{ border-color: #64748b; }}
}}

/* Ensure responsiveness on mobile */
@media (max-width: 768px) {{
    .hero-flex {{
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }}
    .hero-image-col {{
        width: 100%;
        justify-content: center;
    }}
}}
</style>

<div class="hero-flex">
    <div class="hero-image-col">
        <img src="data:image/png;base64,{flower_base64}" style="max-height: 180px; width: auto; max-width: 100%; object-fit: contain; border: none; outline: none; background: transparent; box-shadow: none; -webkit-mask-image: linear-gradient(to right, black 85%, transparent 100%), linear-gradient(to bottom, black 85%, transparent 100%); -webkit-mask-composite: source-in; mask-image: linear-gradient(to right, black 85%, transparent 100%), linear-gradient(to bottom, black 85%, transparent 100%); mask-composite: intersect;" alt="Pebble Hero Graphic" />
    </div>
    <div class="hero-text-col">
        <div class="animated-heading-container">
            <h1>Pebble <span class="tagline-divider">|</span> <span class="typewriter-text">Clear, Grounded Guidance</span></h1>
        </div>
        <div class="hero-container" style="border-bottom: none; margin-bottom: 0; padding-bottom: 0; padding-top: 0; margin-top: 0;">
            <div class="hero-subtitle" style="font-size: 1.2rem; font-style: normal; font-weight: 400; color: #666666; margin-bottom: 0.25rem;">Small questions. Trusted answers.</div>
            <div class="hero-description" style="font-size: 0.95rem; font-weight: 400; color: #444444; line-height: 1.4;">Parenting comes with new questions every day. This AI assistant makes trusted HSE guidance easier to access by turning reliable information from mychild.ie into natural, conversational answers. Get practical support on pregnancy, babies, nutrition, sleep, development, and family wellbeing.</div>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 3.5rem; margin-bottom: 0.5rem; border: none; border-bottom: 3px solid #E2E8F0;' />", unsafe_allow_html=True)

# Data File Path
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "hse_data.txt")

# Load Data
pages = load_hse_data(DATA_FILE)

# Sidebar Configuration
with st.sidebar:
    st.image("https://assets.hse.ie/static/hse-frontend/assets/favicons/favicon-192x192.png", width=64)
    st.markdown("### Settings & Options")
    
    # Read Gemini API Key from environment or .env file
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "your_free_key_here":
        gemini_key = None
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()
        
    st.markdown("---")
    
    # Database stats
    if pages:
        st.markdown(f"**📚 Local Database Stats:**")
        st.markdown(f"- **Pages Crawled:** {len(pages)}")
        chunks = prepare_chunks(pages)
        st.markdown(f"- **Search Index Chunks:** {len(chunks)}")
    else:
        st.markdown("⚠️ **No local data found.** Please run the scraping script first.")

# Process data if loaded
if not pages:
    st.markdown(f"""
    <div class="status-card" style="border-left-color: #e53e3e;">
        <h4 style="color: #e53e3e; margin: 0 0 0.5rem 0;">Database File Not Found</h4>
        <p style="margin: 0; color: #4a5568;">
            The scraped content file <code>{DATA_FILE}</code> could not be found. 
            Please run the crawler script to fetch the HSE My Child pages first:
        </p>
        <pre style="margin-top: 1rem; background: #edf2f7; padding: 0.5rem; border-radius: 4px;"><code>python3 scrape_hse.py</code></pre>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Prepare search index
chunks = prepare_chunks(pages)
vectorizer, tfidf_matrix = build_tfidf_index(chunks)

# Initialize Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History
for message in st.session_state.chat_history:
    role_class = "chat-user" if message["role"] == "user" else "chat-bot"
    st.markdown(f"""
    <div class="chat-bubble {role_class}">
        {message["content"]}
    </div>
    """, unsafe_allow_html=True)

# User Query Input
user_query = st.chat_input("Ask a question about pregnancy, babies, or toddlers...")

if user_query:
    # Display user message immediately
    st.markdown(f'<div class="chat-bubble chat-user">{user_query}</div>', unsafe_allow_html=True)
    
    # Create placeholder for assistant thinking state
    message_placeholder = st.empty()
    
    # Inject skeleton loader HTML and CSS
    skeleton_html = """
    <style>
    .skeleton-wrapper {
        width: 100%;
        padding: 10px 0;
    }
    .loading-text {
        font-family: 'Aspekta', sans-serif;
        font-size: 14px;
        color: #64748b;
        margin-bottom: 12px;
        font-weight: 500;
        animation: pulse-text 2s infinite ease-in-out;
    }
    .skeleton-line {
        height: 16px;
        margin-bottom: 12px;
        border-radius: 4px;
        background: #F0F2F6;
        background-image: linear-gradient(to right, #F0F2F6 0%, #EBF4F8 20%, #F0F2F6 40%, #F0F2F6 100%);
        background-repeat: no-repeat;
        background-size: 800px 100%;
        animation: shimmer 1.5s infinite linear;
    }
    .skeleton-line:last-child {
        width: 60%;
    }
    @keyframes shimmer {
        0% { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }
    @keyframes pulse-text {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    </style>
    <div class="chat-bubble chat-bot">
        <div class="skeleton-wrapper">
            <div class="loading-text">Gathering trusted guidance...</div>
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
        </div>
    </div>
    """
    message_placeholder.markdown(skeleton_html, unsafe_allow_html=True)
    
    # Search for context
    matched_results = search_relevant_chunks(user_query, chunks, vectorizer, tfidf_matrix, top_k=5)
    
    bot_response = ""
    
    if gemini_key:
        # Smart RAG mode using Gemini
        try:
            if not matched_results:
                bot_response = "I'm sorry, I couldn't find official information about that on the My Child pages. Please consult your GP or health nurse for professional medical advice."
            else:
                # Build context prompt
                context_str = ""
                for idx, (chunk, score) in enumerate(matched_results):
                    context_str += f"\n---\nSource {idx+1}: {chunk['title']}\nURL: {chunk['url']}\nContent:\n{chunk['text']}\n---\n"
                
                prompt = f"""
                You are a helpful and expert chatbot representing the HSE My Child service (Ireland's Health Service Executive).
                Your purpose is to answer parent or visitor questions about pregnancy, baby and toddler health, safety, feeding, and milestones.
                
                Strict Rules:
                1. Base your answer ONLY on the official context sections retrieved below. Do not use external health facts that are not present or implied in the context.
                2. If and only if the context does not contain sufficient information to answer the question, state: "I'm sorry, I couldn't find official information about that on the My Child pages. Please consult your GP or health nurse for professional medical advice."
                3. If you can answer the question using the context, do NOT include or append the fallback message under any circumstances.
                
                Context:
                {context_str}
                
                Formatting:
                - Write in a friendly, reassuring, professional tone.
                - Use bullet points, bold text, or numbered lists for readability.
                - At the end of your response, list the matching page titles and URLs under a "Sources & Further Reading:" heading.
                
                User Question: {user_query}
                Answer:
                """
                
                # Call Gemini API
                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                
                bot_response = response.text
            
        except Exception as e:
            error_str = str(e)
            if '429' in error_str:
                bot_response = "Pebble is currently helping a lot of parents right now and needs a quick breather! Please try asking your question again tomorrow, or visit HSE mychild.ie directly for immediate guidance."
            else:
                st.error(error_str)
                bot_response = f"⚠️ **Raw Exception:** {error_str}"
    else:
        # Fallback Local Search mode
        st.sidebar.info("💡 Local Search Mode: Set your GEMINI_API_KEY in the .env file to enable smart conversational AI replies.")
        
        if matched_results:
            bot_response = "Here are the most relevant articles found in the HSE My Child database:\n\n"
            for chunk, score in matched_results[:3]:
                # Format text
                snippet = chunk['text']
                # clean up formatting symbols like heading labels if present
                snippet = re.sub(r'^[A-Z0-9_]+:\s*', '', snippet, flags=re.M)
                
                bot_response += f"""
                <div class="match-card">
                    <div class="match-title">📄 {chunk['title']}</div>
                    <div class="match-url"><a href="{chunk['url']}" target="_blank">{chunk['url']}</a></div>
                    <div class="match-text">{snippet}</div>
                </div>
                """
        else:
            bot_response = "I couldn't find any closely matching sections in the local database. Try rephrasing your search terms."

    # Save to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "bot", "content": bot_response})
    
    # Display final response in placeholder to overwrite skeleton
    final_response = f'<div class="chat-bubble chat-bot">{bot_response}</div>'
    message_placeholder.markdown(final_response, unsafe_allow_html=True)
    st.rerun()
