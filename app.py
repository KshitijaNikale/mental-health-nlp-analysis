import streamlit as st
import joblib
import pickle
import re
import os
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="Mindful — Emotional Assistant", layout="wide", page_icon="💛")

st.markdown(
    """
<style>

:root {
    --bg: #fff8f2;
    --card: #fff2e6;
    --muted: #7a6b63;
    --accent: #ffb997;
    --accent-2: #ffd8b5;
    --shadow: rgba(20,20,20,0.08);
}

/* ---------- APP BG + BASE ---------- */
.stApp {
    background: var(--bg);
    color: #6a5a52;
}

/* ---------- CARDS ---------- */
.card {
    background: var(--card);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(0,0,0,0.03);
    box-shadow: 0 8px 24px var(--shadow);
    transition: 0.18s ease;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 40px rgba(0,0,0,0.10);
}
.card h3 {
    color: #2b2b2b !important;
}

/* ---------- GENERAL TEXT ---------- */
.small, .muted {
    color: var(--muted) !important;
}
.section {
    background: #fffdfa !important;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.04);
}

/* ---------- INPUT + TEXTAREA ---------- */
textarea, input {
    color: #ffffff !important; /* typed text white */
    caret-color: #ffffff !important;
}
textarea::placeholder, input::placeholder {
    color: #e8e1db !important;
}

/* Input/textarea labels */
.stTextInput label,
.stTextArea label {
    color: #5a4a42 !important;
    opacity: 1 !important;
}

/* ---------- INFO + WARNING BOXES ---------- */
.stAlert, .stInfo, .stWarning {
    color: #4a3f39 !important;
}

/* ---------- SECTION HEADERS INSIDE BOXES ---------- */
.section h2 {
    color: #4a3f39 !important;
}

/* ---------- MARKDOWN HEADING FIXES ---------- */
[data-testid="stMarkdown"] h1,
[data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3,
[data-testid="stMarkdown"] h4 {
    color: #4a3f39 !important;
    opacity: 1 !important;
}

/* ---------- SUBHEADER FIX (THE IMPORTANT ONE) ---------- */
/* THIS is what finally fixes: "Write what's on your mind" */
[data-testid="stSubheader"] *,
.stSubheader,
.stSubheader * {
    color: #4a3f39 !important;
    opacity: 1 !important;
}

/* In case Streamlit wraps it inside Markdown container */
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h2 *,
[data-testid="stMarkdownContainer"] h3 * {
    color: #4a3f39 !important;
    opacity: 1 !important;
}
/* Fix helper-message text blending (info, warning, empty prompts) */

/* Info boxes: "Click a card to see..." */
[data-testid="stInfo"],
[data-testid="stInfo"] * {
    color: #5a4a42 !important; /* readable cocoa-brown */
    opacity: 1 !important;
}

/* Warning boxes: "Type something first..." */
[data-testid="stWarning"],
[data-testid="stWarning"] * {
    color: #5a4a42 !important;
    opacity: 1 !important;
}

/* Error/safety/warning line: "If you feel unsafe..." */
[data-testid="stAlert"],
[data-testid="stAlert"] * {
    color: #4a3f39 !important; /* slightly deeper warm brown */
    opacity: 1 !important;
}


</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# SILENT MODEL LOAD (tries common filenames)
# ---------------------------
MODEL = None
VECT = None

def silent_load():
    global MODEL, VECT
    model_files = ["mental_health_model.pkl","model.pkl","mental_health_model.joblib","model.joblib"]
    vect_files = ["tfidf_vectorizer.pkl","tfidf.pkl","vectorizer.pkl","tfidf_vectorizer.joblib"]
    for m in model_files:
        if os.path.exists(m):
            try:
                MODEL = joblib.load(m)
                break
            except Exception:
                try:
                    with open(m,"rb") as f:
                        MODEL = pickle.load(f)
                        break
                except Exception:
                    continue
    for v in vect_files:
        if os.path.exists(v):
            try:
                VECT = joblib.load(v)
                break
            except Exception:
                try:
                    with open(v,"rb") as f:
                        VECT = pickle.load(f)
                        break
                except Exception:
                    continue

silent_load()

# ---------------------------
# SAFE FALLBACK (used silently if no model)
# ---------------------------
ANX_WORDS = {"panic","anxious","scared","worried","overthinking","nervous","fear","shaky"}
DEP_WORDS = {"sad","empty","tired","hopeless","worthless","numb","alone","lost"}
SUI_WORDS = {"suicide","kill myself","end my life","die","cant go on","no reason to live"}

def fallback_predict(text):
    t = clean(text)
    tokens = set(t.split())
    a = len(tokens & ANX_WORDS)
    d = len(tokens & DEP_WORDS)
    s = len(tokens & SUI_WORDS)
    scores = {"Anxiety": a, "Depression": d, "Suicide": s}
    top = max(scores, key=scores.get)
    if sum(scores.values()) == 0:
        return "Calm / Neutral", {"Anxiety":0.0,"Depression":0.0,"Suicide":0.0}
    total = sum(scores.values())
    probs = {k: round(v/total,2) for k,v in scores.items()}
    return top, probs

# ---------------------------
# TEXT CLEANER & METRICS
# ---------------------------
def clean(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = re.sub(r"[^a-zA-Z\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

POS = {"good","better","calm","okay","relief","hope","safe","well","ok","fine"}
NEG = {"sad","tired","hopeless","worthless","numb","suicide","die","hurt","alone"}

def input_metrics(text):
    t = clean(text)
    words = t.split()
    wc = len(words)
    pos = sum(1 for w in words if w in POS)
    neg = sum(1 for w in words if w in NEG)
    polarity_raw = (pos - neg) / max(1, wc)
    # human-friendly tone
    if polarity_raw > 0.15:
        tone = "Mostly positive"
    elif polarity_raw < -0.15:
        tone = "Mostly negative"
    else:
        tone = "Mixed / Neutral"
    neg_density = round(neg / max(1, wc), 2)
    intensity = min(wc / 120, 1.0)
    return {"word_count": wc, "tone": tone, "neg_density": neg_density, "intensity": round(intensity,2)}

# ---------------------------
# MODEL PREDICTION (silent usage)
# ---------------------------
def predict_with_model(text):
    cleaned = clean(text)
    if MODEL is None or VECT is None:
        return None
    try:
        vec = VECT.transform([cleaned])
        if hasattr(MODEL, "predict_proba"):
            probs_arr = MODEL.predict_proba(vec)[0]
            classes = [str(c) for c in MODEL.classes_]
            # map class names to our keys where possible
            probs = {"Anxiety":0.0,"Depression":0.0,"Suicide":0.0}
            for i,c in enumerate(classes):
                lc = c.lower()
                if "anx" in lc: key="Anxiety"
                elif "depress" in lc: key="Depression"
                elif "sui" in lc: key="Suicide"
                else: key = c
                probs[key] = float(probs_arr[i])
            return probs
        else:
            pred = MODEL.predict(vec)[0]
            pred = str(pred)
            probs = {"Anxiety":0.0,"Depression":0.0,"Suicide":0.0}
            if "anx" in pred.lower(): probs["Anxiety"]=1.0
            elif "depress" in pred.lower(): probs["Depression"]=1.0
            elif "sui" in pred.lower(): probs["Suicide"]=1.0
            return probs
    except Exception:
        return None

# ---------------------------
# HUMAN-FRIENDLY TEXTS & ROUTINES
# ---------------------------
ROUTINES = {
    "Anxiety": [
        "Try the 4-4-6 breathing for 2 minutes (inhale 4s, hold 4s, exhale 6s).",
        "Ground yourself: name 5 things you can see and 4 things you can touch.",
        "Sip water slowly and soften your shoulders."
    ],
    "Depression": [
        "Sit near some sunlight for 5 minutes.",
        "Do one tiny task (fill a glass, or open a window).",
        "Listen to a gentle song you like; try a short walk."
    ],
    "Suicide": [
        "You’re not alone. Call a trusted person now, or local emergency services if in danger.",
        "Stay with someone if you can. Ground with steady breaths.",
        "If you feel unsafe, please reach out to emergency help right away."
    ],
    "Calm / Neutral": [
        "You're okay for now — small self-care: water, fresh air, stretch."
    ]
}

MESSAGES = {
    "Anxiety": "Your words show tension and worry. Start with a grounding breath. You’re safe in this moment.",
    "Depression": "There’s a heaviness in your words. Small, gentle actions can slowly help — you matter.",
    "Suicide": "This looks like crisis language. If you are in danger or thinking about harming yourself, contact local emergency services or a crisis hotline immediately.",
    "Calm / Neutral": "Your message seems steady. Keep checking in with small self-care."
}

# ---------------------------
# Clinical labels + supportive messages (lists)
# ---------------------------
clinical_depression = [
    ("Self-Blame", "You’re carrying weight that isn’t yours. That’s self-blaming, not truth."),
    ("Anhedonia", "Nothing feels pleasurable lately — that’s anhedonia, not a sign you’re broken."),
    ("Rumination", "You’re looping the same thoughts — that’s rumination, not reality."),
    ("Cognitive Fatigue", "Your thinking feels heavy because your brain is tired — that’s cognitive fatigue."),
    ("Hopelessness Bias", "Your mind is filtering out brighter possibilities — that’s a bias, not destiny.")
]

clinical_anxiety = [
    ("Catastrophizing", "Your mind is jumping to worst-case scenarios — that’s fear, not fact."),
    ("Threat Sensitivity", "Everything feels dangerous right now — that’s heightened threat sensitivity."),
    ("Hypervigilance", "You’re scanning for danger nonstop — that’s hypervigilance, not intuition."),
    ("Racing Thoughts", "Your thoughts are racing — that’s overload, not failure."),
    ("Uncertainty Intolerance", "Not knowing is hard — that’s anxiety, not a personal flaw.")
]

clinical_anger = [
    ("Emotional Flooding", "You’re overwhelmed — that’s emotional flooding, not danger."),
    ("Frustration Overload", "Your system is overloaded; that heat is frustration, not failure."),
    ("Cognitive Narrowing", "Anger narrows focus — that’s a reaction, not a choice."),
    ("Boundary Trigger", "This fire often comes from a crossed boundary, not because you’re 'too much'."),
    ("Suppressed Resentment", "This may be resentment built up from feeling unheard, not uncontrollable rage.")
]

clinical_fear = [
    ("Freeze Response", "Your mind is freezing to protect you — not abandoning you."),
    ("Sense of Overwhelm", "You feel swamped — that’s overwhelm, not reality collapsing."),
    ("Safety Seeking", "You’re looking for escape routes — that’s fear, not failure."),
    ("Future Threat Projection", "You’re imagining threats ahead — that’s projection, not prophecy."),
    ("Emotional Shock", "Your body is stunned — that’s shock, not brokenness.")
]

clinical_positive = [
    ("Stable Grounding", "You’re steady right now — hold this space."),
    ("Emotional Clarity", "Your mind feels clearer — trust that clarity."),
    ("Adaptive Thinking", "You’re responding with balance — that’s a healthy pattern."),
    ("Healthy Regulation", "You’re regulating well — keep honoring your pace."),
    ("Resilience Mode", "This calm is resilience showing through.")
]

clinical_labels = {
    "Depression": clinical_depression,
    "Anxiety": clinical_anxiety,
    "Anger": clinical_anger,
    "Fear / Stress": clinical_fear,
    "Calm / Neutral": clinical_positive
}

def get_clinical_message(detected_label):
    # Map detected_label string to clinical_labels keys
    if not detected_label:
        key = "Calm / Neutral"
    else:
        dl = detected_label.lower()
        if "depress" in dl:
            key = "Depression"
        elif "anx" in dl:
            key = "Anxiety"
        elif "sui" in dl:
            key = "Suicide" # no dedicated suicide list; we'll fallback
        elif "ang" in dl or "rage" in dl:
            key = "Anger"
        elif "fear" in dl or "stress" in dl:
            key = "Fear / Stress"
        elif "calm" in dl or "neutral" in dl:
            key = "Calm / Neutral"
        else:
            key = "Calm / Neutral"

    # If suicide or unknown, fallback to depression/anxiety combos
    if key == "Suicide":
        # choose from depression or anxiety lists for supportive messaging
        pool = clinical_depression + clinical_anxiety
    else:
        pool = clinical_labels.get(key, clinical_positive)

    label, msg = random.choice(pool)
    return label, msg

# ---------------------------
# UI: Header + input
# ---------------------------
st.markdown("<div style='padding:10px 0'><h1 style='margin:0;color:#5b3a29'>💛 Mindful — Emotions Analyzer</h1><div class='small muted' style='margin-top:6px'>A calm, private place to check your feelings. Your words stay in this session.</div></div>", unsafe_allow_html=True)
st.markdown("---")

st.session_state.setdefault("selected", None)
st.session_state.setdefault("history", [])

# Input
st.subheader("Write what’s on your mind")
user_text = st.text_area("", height=140, placeholder="Type a diary entry, a message, or how you feel...")

# Card row (click to open)
cols = st.columns(4, gap="large")
with cols[0]:
    if st.button("🔮 Detected Emotion", key="c1"):
        st.session_state.selected = "emotion"
with cols[1]:
    if st.button("📊 How your mind sounds", key="c2"):
        st.session_state.selected = "breakdown"
with cols[2]:
    if st.button("🔎 Emotional Insights", key="c3"):
        st.session_state.selected = "insights"
with cols[3]:
    if st.button("🌿 Routine & Support", key="c4"):
        st.session_state.selected = "routine"

st.markdown("") # spacer

# ---------------------------
# PROCESS INPUT HIDDENLY
# ---------------------------
detected = None
probs = {"Anxiety":0.0,"Depression":0.0,"Suicide":0.0}
if user_text.strip():
    probs = predict_with_model(user_text)
    if probs is None:
        label, fallback_probs = fallback_predict(user_text)
        probs = fallback_probs
        detected = label
    else:
        detected = max(probs, key=probs.get)
else:
    detected = None

# compute emotion score and breakdown for storage
emotion_score = round(max(probs.values()) if probs else 0.0, 2)
breakdown = {k.lower(): float(v) for k, v in probs.items()}

# clinical label + message for this detection
clinical_label, clinical_message = get_clinical_message(detected)

# small insight text (reuse clinical message as insight)
insight = clinical_message

# store to history when analyze-like click happens
def save_history_entry():
    if not user_text.strip():
        return
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "time": timestamp,
        "user_text": user_text,
        "detected": detected or "Calm / Neutral",
        "emotion_score": emotion_score,
        "breakdown": breakdown,
        "insight_label": clinical_label,
        "insight_message": clinical_message,
        "routine": ROUTINES.get(detected, ROUTINES["Calm / Neutral"]),
    }
    # avoid duplicate if same text is already top
    if not st.session_state["history"] or st.session_state["history"][0]["user_text"] != user_text:
        st.session_state["history"].insert(0, entry)

# ---------------------------
# CARD CONTENTS (human phrasing)
# ---------------------------
if st.session_state["selected"] is None:
    st.info("Click a card to see a calm, simple readout of your text. We are here to Help you :))")
else:
    if not user_text.strip():
        st.warning("Type something first, then click a card.")
    else:
        # compute human metrics
        metrics = input_metrics(user_text)
        # save history entry
        save_history_entry()

        # Emotion card
        if st.session_state["selected"] == "emotion":
            st.markdown(
                f"<div class='section'><h2 style='margin-bottom:6px'>Detected feeling</h2>"
                f"<div style='font-size:20px;padding:10px;border-radius:10px;background:linear-gradient(90deg,#ffd8b5,#ffb997);display:inline-block'>{detected}</div>"
                f"<p class='small' style='margin-top:12px'>{MESSAGES.get(detected,'If you feel distressed, reach out to someone you trust.')}</p></div>",
                unsafe_allow_html=True
            )

        # Breakdown (friendly bars + sentence)
        if st.session_state["selected"] == "breakdown":
            st.markdown("<div class='section'><h2 style='margin-bottom:6px'>How your message sounds</h2></div>", unsafe_allow_html=True)
            dfb = pd.DataFrame({"Emotion":["Anxiety","Depression","Suicide"], "Score":[probs.get("Anxiety",0),probs.get("Depression",0),probs.get("Suicide",0)]})
            st.bar_chart(dfb.set_index("Emotion"))
            top = dfb.loc[dfb["Score"].idxmax()]["Emotion"]
            st.write(f"Overall, this message reads most like **{top}**.")
            st.write(f"Emotional tone: **{metrics['tone']}** • Intensity: **{int(metrics['intensity']*100)}%**")

        # Insights (human-language)
        if st.session_state["selected"] == "insights":
            st.markdown("<div class='section'><h2 style='margin-bottom:6px'>Quick insights</h2></div>", unsafe_allow_html=True)
            st.write(f"- **Word count:** {metrics['word_count']}")
            st.write(f"- **Tone:** {metrics['tone']}")
            if metrics['neg_density'] > 0.18:
                st.write("- **Note:** The message has several stress-related words. Consider grounding.")
            else:
                st.write("- **Note:** The message doesn't show a high density of distress words.")
            st.write("")
            st.write("If you like, try writing how this situation started — sometimes context reduces uncertainty.")

        # Routine & Support
        if st.session_state["selected"] == "routine":
            st.markdown("<div class='section'><h2 style='margin-bottom:6px'>Gentle routine for right now</h2></div>", unsafe_allow_html=True)
            steps = ROUTINES.get(detected, ROUTINES["Calm / Neutral"])
            for s in steps:
                st.write("• " + s)
            st.write("")
            st.info("If you feel unsafe at any point, please contact local emergency services or a trusted person immediately.")

            # Clinical label + single supportive sentence (as requested)
            st.markdown("### 🧠 Clinical Insight")
            st.write(f"**{clinical_label}:** {clinical_message}")

# ---------------------------
# bottom: session history (small)
# ---------------------------
st.markdown("---")
st.markdown("**Recent checks (this session)**")
if st.session_state["history"]:
    for h in st.session_state["history"][:6]:
        st.markdown(f"- [{h['time']}] **{h['detected']}** — {h['user_text'][:80]}{'...' if len(h['user_text'])>80 else ''}")
else:
    st.markdown("_You haven't checked anything yet — everything stays in this browser session._")

# ---- FULL SESSION EXPORT (DETAILED) ----
if st.session_state["history"]:

    export_text = "Mindful Session Report\n"
    export_text += "=======================\n\n"

    for i, h in enumerate(st.session_state["history"], 1):
        export_text += f"Entry {i}\n"
        export_text += f"Time: {h['time']}\n"
        export_text += f"User Text: {h['user_text']}\n\n"

        export_text += f"Detected Emotion: {h['detected']}\n"
        export_text += f"Emotion Score: {h.get('emotion_score', '')}\n\n"

        # breakdown (dictionary)
        export_text += "Breakdown:\n"
        for k, v in h.get("breakdown", {}).items():
            export_text += f" - {k}: {v}\n"
        export_text += "\n"

        export_text += f"Insight Label: {h.get('insight_label','')}\n"
        export_text += f"Insight Message: {h.get('insight_message','')}\n\n"

        export_text += "Routine & Support:\n"
        for step in h.get("routine", []):
            export_text += f" • {step}\n"
        export_text += "\n"

        export_text += "----------------------------\n\n"

    st.download_button(
        label="📥 Download Full Session (Detailed)",
        data=export_text,
        file_name="mindful_full_session.txt",
        mime="text/plain",
    )
