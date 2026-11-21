import os
import re
from urllib.parse import quote
import pandas as pd
import requests
import streamlit as st
# Optional ML signal (only used if scikit-learn and data/demo_news.csv exist)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Fact-Check Finder", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
    .verdict-box {padding: 1.1rem 1.3rem; border-radius: 12px; margin-bottom: 1rem; border-left: 8px solid; font-size: 1.05rem;}
    .v-false {background:#fdecea; border-color:#d93025; color:#7a1610;}
    .v-true {background:#e6f4ea; border-color:#188038; color:#0d4a1f;}
    .v-mixed {background:#fef7e0; border-color:#f9ab00; color:#6b4a00;}
    .v-none {background:#eef1f5; border-color:#5f6368; color:#2b2f33;}
    .review-card {border:1px solid #dadce0; border-radius:10px; padding:.8rem 1rem; margin-bottom:.7rem; background:#ffffff; color:#202124;}
    .badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:600; margin-left:6px;}
    .b-false {background:#d93025; color:white;}
    .b-true {background:#188038; color:white;}
    .b-mixed {background:#f9ab00; color:black;}
    .b-none {background:#5f6368; color:white;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📰 Fact-Check Finder")
st.caption(
    "Search the claim in Google's Fact Check database, see what professional "
    "fact-checkers concluded, and check the text for misleading writing style."
)

# --------------------------------------------------------------------------
# Sidebar: API key + settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = ""
    try:
        api_key = st.secrets.get("GOOGLE_FACTCHECK_API_KEY", "")
    except Exception:
        pass
    api_key = api_key or os.getenv("GOOGLE_FACTCHECK_API_KEY", "")

    if not api_key:
        api_key = st.text_input(
            "Google Fact Check API key",
            type="password",
            help="Free key from Google Cloud Console → enable 'Fact Check Tools API'.",
        )
    else:
        st.success("API key loaded ✅")

    language = st.selectbox(
        "Claim language",
        ["en", "hi", "ml", "ta", "te", "bn", "mr", "gu", "kn", "ur", "es", "fr", "de", "pt", "ar"],
        index=0,
        help="Fact-checks are filtered by this language code.",
    )
    max_age = st.slider("Only fact-checks from the last (days)", 0, 3650, 0, help="0 = no limit")
    page_size = st.slider("Max results", 5, 30, 15)

    with st.expander("How to get a free API key"):
        st.markdown(
            """
            1. Open [Google Cloud Console](https://console.cloud.google.com/).
            2. Create a project → **APIs & Services → Library**.
            3. Enable **Fact Check Tools API**.
            4. **Credentials → Create credentials → API key**.
            5. Save it in `.streamlit/secrets.toml`:
            ```toml
            GOOGLE_FACTCHECK_API_KEY = "your-key-here"
            ```
            """
        )

# --------------------------------------------------------------------------
# Rating interpretation
# --------------------------------------------------------------------------
FALSE_WORDS = [
    "false", "fake", "pants on fire", "misleading", "incorrect", "hoax", "scam", 
    "fabricated", "satire", "no evidence", "not true", "untrue", "baseless", 
    "manipulated", "doctored", "misattributed", "wrong", "mostly false", "distorted", 
    "made up", "fiction", "debunked", "altered", "bogus", "clickbait", "deepfake", "misinformation", "falso",
]
TRUE_WORDS = [
    "true", "correct", "accurate", "right", "confirmed", "legit", "genuine", 
    "verified", "mostly true", "factual", "real", "cierto", "verdadero",
]
MIXED_WORDS = [
    "mixture", "mixed", "half", "partly", "partially", "missing context", "needs context", 
    "unproven", "unverified", "exaggerat", "outdated", "out of context", "lacks context", 
    "unclear", "disputed", "misleading context", "some truth", "cherry",
]

def classify_rating(rating: str) -> str:
    """Return 'false', 'true', 'mixed' or 'none' for a textual rating."""
    r = (rating or "").lower().strip()
    if not r:
        return "none"
    if any(w in r for w in MIXED_WORDS) and "mostly false" not in r:
        return "mixed"
    if any(w in r for w in FALSE_WORDS):
        return "false"
    if any(re.search(rf"\b{re.escape(w)}\b", r) for w in TRUE_WORDS):
        return "true"
    return "none"

BADGE = {"false": "b-false", "true": "b-true", "mixed": "b-mixed", "none": "b-none"}
LABEL = {"false": "FALSE / MISLEADING", "true": "TRUE", "mixed": "MIXED / NEEDS CONTEXT", "none": "UNCLASSIFIED"}

# --------------------------------------------------------------------------
# Google Fact Check Tools API
# --------------------------------------------------------------------------
API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

@st.cache_data(ttl=3600, show_spinner=False)
def search_fact_checks(query: str, key: str, lang: str, max_age_days: int, size: int):
    params = {"query": query, "key": key, "pageSize": size}
    if lang:
        params["languageCode"] = lang
    if max_age_days:
        params["maxAgeDays"] = max_age_days
    try:
        resp = requests.get(API_URL, params=params, timeout=15)
    except requests.RequestException as exc:
        return {"error": f"Network error: {exc}"}
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        return {"error": f"API error {resp.status_code}: {msg}"}
    return resp.json()

def build_query(text: str) -> str:
    """Fact-check search works best with a short, focused claim."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= 200:
        return text
    sentences = re.split(r"(?<=([.!?]))\s+", text)
    out = ""
    for s in sentences:
        if len(out) + len(s) > 200:
            break
        out += (" " if out else "") + s
    return (out or text[:200]).strip()

def flatten_reviews(data: dict):
    rows = []
    for claim in data.get("claims", []):
        for rev in claim.get("claimReview", []):
            rating = rev.get("textualRating", "")
            rows.append(
                {
                    "claim": claim.get("text", ""),
                    "claimant": claim.get("claimant", "Unknown"),
                    "claim_date": (claim.get("claimDate") or "")[:10],
                    "publisher": rev.get("publisher", {}).get("name", "Unknown"),
                    "site": rev.get("publisher", {}).get("site", ""),
                    "title": rev.get("title", "Fact-check article"),
                    "url": rev.get("url", ""),
                    "rating": rating,
                    "verdict": classify_rating(rating),
                    "review_date": (rev.get("reviewDate") or "")[:10],
                }
            )
    return rows

def overall_verdict(rows):
    counts = {"false": 0, "true": 0, "mixed": 0, "none": 0}
    for r in rows:
        counts[r["verdict"]] += 1
    classified = counts["false"] + counts["true"] + counts["mixed"]
    if classified == 0:
        return "none", counts
    if counts["false"] > counts["true"] and counts["false"] >= counts["mixed"]:
        return "false", counts
    if counts["true"] > counts["false"] and counts["true"] >= counts["mixed"]:
        return "true", counts
    return "mixed", counts

# --------------------------------------------------------------------------
# Writing-style analysis (rule-based, explainable)
# --------------------------------------------------------------------------
CLICKBAIT = [
    "you won't believe", "shocking", "they don't want you to know", "share before", 
    "before it's deleted", "100% proven", "miracle", "secret", "exposed", "bombshell", 
    "doctors hate", "breaking!!!", "urgent", "forward this", "share this", "mainstream media", 
    "wake up", "cover-up", "cover up", "banned", "leaked", "must watch", "going viral", 
    "hidden truth", "cure for", "guaranteed",
]
SOURCE_HINTS = [
    "according to", "said", "says", "reported", "report", "study", "published", 
    "official", "spokesperson", "statement", "ministry", "police", "court", 
    "researchers", "university", "data from",
]

def style_signals(text: str):
    signals = []
    score = 0
    words = re.findall(r"[A-Za-z']+", text)
    caps_words = [w for w in words if len(w) > 3 and w.isupper()]
    caps_ratio = len(caps_words) / max(len(words), 1)
    if caps_ratio > 0.15:
        score += 2
        signals.append(f"Lots of ALL-CAPS words ({caps_ratio:.0%}) – typical of emotional pressure.")
    
    excls = text.count("!")
    if excls >= 2:
        score += 2
        signals.append(f"{excls} exclamation marks.")
    
    hits = [p for p in CLICKBAIT if p in text.lower()]
    if hits:
        score += min(len(hits), 3) * 2
        signals.append("Clickbait / pressure phrases: " + ", ".join(f"“{h}”" for h in hits[:5]))
    
    if not any(h in text.lower() for h in SOURCE_HINTS):
        score += 1
        signals.append("No named source, official, study or document is mentioned.")
    
    if re.search(r"\b(always|never|everyone|nobody|100%|all of them)\b", text, re.I):
        score += 1
        signals.append("Absolute language (always / never / everyone / 100%).")
    
    if re.search(r"\b(share|forward|send)\b.{0,30}\b(now|everyone|friends|groups?)\b", text, re.I):
        score += 2
        signals.append("Asks you to share/forward urgently.")
        
    level = "high" if score >= 5 else "medium" if score >= 3 else "low"
    return level, score, signals

@st.cache_resource
def load_optional_model():
    if not SKLEARN_OK or not os.path.exists("data/demo_news.csv"):
        return None
    data = pd.read_csv("data/demo_news.csv")
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]
    )
    model.fit(data["text"], data["label"])
    return model

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
tab_check, tab_how = st.tabs(["🔍 Check a claim", "ℹ️ How it works"])

with tab_check:
    left, right = st.columns([3, 2], gap="large")
    with left:
        news = st.text_area(
            "Paste a headline, WhatsApp forward or claim",
            height=170,
            placeholder="e.g. Drinking hot water cures COVID-19",
        )
        run = st.button("🔎 Check this claim", type="primary", use_container_width=True)
    with right:
        st.markdown("**Tips for better results**")
        st.markdown(
            "- Use the **core claim**, not the whole article.\n"
            "- Short, specific sentences match best.\n"
            "- Pick the correct **language** in the sidebar.\n"
            "- No result ≠ true. It may simply not be checked yet."
        )

    if run:
        text = news.strip()
        if len(text) < 10:
            st.warning("Please enter at least 10 characters.")
        else:
            query = build_query(text)
            st.markdown(f"**Searching for:** _{query}_")
            rows, error = [], None
            
            if not api_key:
                error = "No API key set. Add your Google Fact Check API key in the sidebar."
            else:
                with st.spinner("Searching Google Fact Check database..."):
                    data = search_fact_checks(query, api_key, language, max_age, page_size)
                    if "error" in data:
                        error = data["error"]
                    else:
                        rows = flatten_reviews(data)

            if error:
                st.error(error)

            # ---------- Fact-check verdict ----------
            st.subheader("1️⃣ What fact-checkers say")
            if rows:
                verdict, counts = overall_verdict(rows)
                css = {"false": "v-false", "true": "v-true", "mixed": "v-mixed", "none": "v-none"}[verdict]
                headline = {
                    "false": "🚩 Fact-checkers mostly rated similar claims <b>FALSE or MISLEADING</b>.",
                    "true": "✅ Fact-checkers mostly rated similar claims <b>TRUE</b>.",
                    "mixed": "⚠️ Fact-checkers gave <b>mixed / context-dependent</b> ratings.",
                    "none": "ℹ️ Fact-checks were found but ratings could not be classified. Read them below.",
                }[verdict]
                
                st.markdown(f'<div class="verdict-box {css}">{headline}</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("False / misleading", counts["false"])
                c2.metric("True", counts["true"])
                c3.metric("Mixed", counts["mixed"])
                c4.metric("Total reviews", len(rows))
                
                st.caption("Matches are for *similar* claims. Read each article to confirm it is the same claim you are checking.")
                
                for r in rows:
                    st.markdown(
                        f"""
                        <div class="review-card">
                        <b>{r['publisher']}</b> <span class="badge {BADGE[r['verdict']]}">{r['rating'] or 'No rating'}</span><br>
                        <i>Claim:</i> {r['claim'][:300]}<br>
                        <small>By: {r['claimant']} · Reviewed: {r['review_date'] or 'n/a'}</small><br>
                        <a href="{r['url']}" target="_blank">📄 {r['title']}</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                df = pd.DataFrame(rows)[["publisher", "rating", "claim", "url", "review_date"]]
                st.download_button(
                    "⬇️ Download results (CSV)",
                    df.to_csv(index=False).encode("utf-8"),
                    "fact_check_results.csv",
                    "text/csv",
                )
            elif not error:
                st.markdown(
                    '<div class="verdict-box v-none">🔍 <b>No matching fact-check found.</b> '
                    'This does <u>not</u> mean the claim is true. It may be new, local, or not yet reviewed.</div>',
                    unsafe_allow_html=True,
                )

            # ---------- Style analysis ----------
            st.subheader("2️⃣ Writing-style warning signs")
            level, score, signals = style_signals(text)
            color = {"high": "🔴", "medium": "🟠", "low": "🟢"}[level]
            st.markdown(f"{color} **Misleading-style risk: {level.upper()}** (score {score})")
            
            if signals:
                for s in signals:
                    st.markdown(f"- {s}")
            else:
                st.markdown("- No obvious clickbait or pressure patterns found.")
            st.caption("Style only shows how the text is written, not whether the facts are correct.")

            model = load_optional_model()
            if model is not None and len(text) >= 20:
                with st.expander("Experimental: ML text-pattern model (tiny demo data)"):
                    label = model.predict([text])[0]
                    probs = model.predict_proba([text])[0]
                    conf = float(probs[list(model.classes_).index(label)]) * 100
                    st.write(f"Closest pattern: **{label}** ({conf:.1f}%)")
                    st.caption("Trained on a very small dataset, so treat this as a curiosity only.")

            # ---------- Manual verification ----------
            st.subheader("3️⃣ Verify it yourself")
            q = quote(query)
            b1, b2, b3, b4 = st.columns(4)
            b1.link_button("Google Fact Check Explorer", f"https://toolbox.google.com/factcheck/explorer/search/{q}", use_container_width=True)
            b2.link_button("Google Search", f"https://www.google.com/search?q={q}+fact+check", use_container_width=True)
            b3.link_button("Google News", f"https://news.google.com/search?q={q}", use_container_width=True)
            b4.link_button("Reverse image search", "https://lens.google.com/", use_container_width=True)

            for i, item in enumerate(
                [
                    "Find the original publisher and read the full article.",
                    "Check the date – is this old news shared as new?",
                    "Look for named sources, documents or official statements.",
                    "Confirm with at least two independent reputable outlets.",
                    "Check fact-checkers (BOOM, Alt News, Vishvas News, AFP, Reuters, PolitiFact).",
                    "If it has an image/video, run a reverse image search.",
                    "Don't share if you can't verify it.",
                ]
            ):
                st.checkbox(item, key=f"verify_{i}")

with tab_how:
    st.markdown(
        """
        ### How this app works
        **1. Google Fact Check Tools API** – the same database used by Google Search's "Fact check" labels and the Fact Check Explorer. Publishers like Reuters, AFP, BOOM, Alt News and PolitiFact tag their articles with *ClaimReview* markup, and the API lets us search them. The app reads each textual rating (e.g. "False", "Mostly True", "Missing context") and summarises it.
        
        **2. Writing-style analysis** – transparent rules that look for ALL-CAPS, exclamation marks, clickbait phrases, absolute words, urgent "share now" requests and missing sources.
        
        **3. Manual verification links** – quick links to search and verify further.
        
        ### Limitations
        - A claim with **no** fact-check is *unverified*, not *true*.
        - Matches are by similarity; always open the article and confirm it's the same claim.
        - Style warnings can't judge facts. Real news can be badly written and fake news can look calm.
        - Ratings are classified by keywords; wording differs between publishers.
        """
    )
