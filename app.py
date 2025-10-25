import os
import json
from typing import List, Tuple

from dotenv import load_dotenv
import streamlit as st

# Google GenAI
from google import genai
from google.genai import types

# RAG helpers
import numpy as np
import pandas as pd
# from PyPDF2 import PdfReader
from pypdf import PdfReader

# ---------- Setup ----------
load_dotenv()  # loads GEMINI_API_KEY from .env (if present)
client = genai.Client()  # picks up GEMINI_API_KEY automatically

MODEL_NAME = "gemini-2.5-flash"  # fast + free-tier friendly
EMBED_MODEL = "text-embedding-004"

st.set_page_config(page_title="ER Pre-Op Assistant (SIM) + Chat", page_icon="🤖")

# ---------- Sidebar: Modes & Controls ----------
with st.sidebar:
    st.header("Mode")
    mode = st.selectbox("Select assistant mode", ["General Chat", "ER Pre-Op Assistant (SIM)"])

    st.divider()
    st.subheader("Persona / Controls")
    if mode == "General Chat":
        persona = st.text_area(
            "System / Persona",
            value="You are a concise, helpful assistant.",
            height=120,
        )
        temp = st.slider("Creativity (temperature)", 0.0, 2.0, 0.7, 0.1)
        top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.05)
    else:
        persona = (
            "You are an ER Pre-Operative Readiness Assistant used for SIMULATION ONLY. "
            "You assist licensed clinicians by producing a structured PRE-OP CHECKLIST "
            "for considerations PRIOR to surgery.\n"
            "Rules:\n"
            "• Base answers ONLY on the provided PATIENT REGISTRY context and the user's question.\n"
            "• If required data is missing in the registry, say 'Insufficient data in registry' and ask for it.\n"
            "• Never give medication doses or invasive treatment orders; suggest consulting appropriate specialists.\n"
            "• Always highlight RED FLAGS, CONTRAINDICATIONS, ANTICOAGULATION, ALLERGIES, INFECTIOUS RISK, IMPLANTS/DEVICES.\n"
            "• Include a short rationale under each checklist item and cite the registry snippet name(s) you used.\n"
            "• This is NOT medical advice and is for training/simulation purposes only."
        )
        temp = 0.2
        top_p = 0.9

    st.divider()
    if st.button("Reset conversation"):
        st.session_state.messages = [
            {"role": "model", "content": "New session started. How can I help?"}
        ]
        st.rerun()

# ---------- Header / Banner ----------
if mode == "ER Pre-Op Assistant (SIM)":
    st.title("🏥 ER Pre-Op Assistant (Simulation)")
    st.warning("SIMULATION ONLY • Do not use with real patient data • Not medical advice.")
else:
    st.title("🤖 *Insert cool name here*")

# ---------- Knowledge Base (ER mode): Upload & Embed ----------
KB_TEXTS: List[str] = []
KB_META: List[dict] = []
EMB = None  # np.ndarray

def _read_file_to_text(file) -> str:
    if file.type == "application/pdf":
        try:
            reader = PdfReader(file)
            return "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    elif file.type in ("text/plain",):
        return file.read().decode("utf-8", errors="ignore")
    elif file.type in ("application/json",):
        try:
            j = pd.read_json(file)
            return j.to_csv(index=False)
        except Exception:
            return file.read().decode("utf-8", errors="ignore")
    elif file.type in ("application/vnd.ms-excel", "text/csv", "application/csv"):
        try:
            df = pd.read_csv(file)
            return df.to_csv(index=False)
        except Exception:
            return file.read().decode("utf-8", errors="ignore")
    else:
        return ""

def _chunk(text: str, chunk_size: int = 1200) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[{"parts": [{"text": t}]} for t in texts],
    )
    vecs = [e.values for e in result.embeddings]
    return np.asarray(vecs, dtype=np.float32)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T

def _retrieve(query: str, k: int = 6) -> List[Tuple[str, str, float]]:
    """Returns list of (chunk_text, source_name, similarity)."""
    if EMB is None or len(KB_TEXTS) == 0:
        return []
    qv = _embed_texts([query])
    sims = _cosine_sim(qv, EMB)[0]
    idx = np.argsort(-sims)[:k]
    return [(KB_TEXTS[i], KB_META[i]["source"], float(sims[i])) for i in idx]

# Upload only shown/used in ER mode
if mode == "ER Pre-Op Assistant (SIM)":
    with st.expander("Upload synthetic patient registry (CSV, JSON, TXT, or PDF)"):
        uploads = st.file_uploader(
            "Upload one or more files",
            type=["csv", "json", "txt", "pdf"],
            accept_multiple_files=True
        )
        if uploads:
            for f in uploads:
                text = _read_file_to_text(f)
                if not text:
                    continue
                for c in _chunk(text, 1200):
                    KB_TEXTS.append(c)
                    KB_META.append({"source": f.name})
            if KB_TEXTS:
                with st.spinner("Indexing registry…"):
                    EMB = _embed_texts(KB_TEXTS)
                st.success(f"Indexed {len(KB_TEXTS)} chunks from {len(set(m['source'] for m in KB_META))} file(s).")

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "model", "content": "Hi! I'm the coolest LLM around. What's up?"}
    ]

# Render history
for m in st.session_state.messages:
    with st.chat_message("assistant" if m["role"] == "model" else "user"):
        st.markdown(m["content"])

# ---------- Generation helpers ----------
SCHEMA_ER_JSON = """
Return ONLY valid JSON with this exact structure:

{
  "patient_match": "best-guess name or 'unknown'",
  "chief_concern": "short string",
  "pre_op_checklist": [
    {
      "section": "Airway|Neuro|Cardio|Resp|Renal|Heme|Infectious|Implants|Allergies|Meds|Anticoagulation|Consent|Imaging/Labs|Consults|Other",
      "item": "what to check or confirm",
      "rationale": "why this matters",
      "evidence": ["registry_snippet_1", "registry_snippet_2"]
    }
  ],
  "red_flags": [
    {"item":"critical risk", "rationale":"why", "evidence":["registry_snippet"] }
  ],
  "contraindications": [
    {"item":"contraindication", "rationale":"why", "evidence":["registry_snippet"] }
  ],
  "missing_information_requests": [
    "Ask for these key missing items if not in registry..."
  ],
  "disclaimer": "This is a simulation and not medical advice."
}
""".strip()

def _run_general(history, user_prompt):
    return client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=history + [{"role": "user", "parts": [{"text": user_prompt}]}],
        config=types.GenerateContentConfig(
            system_instruction=persona,
            temperature=float(temp),   # sliders may be Decimal; coerce to float
            top_p=float(top_p),
            # max_output_tokens=1024,  # optional
            # stop_sequences=["<END>"],# optional
        ),
    )

def _run_er(history, user_prompt):
    ctx = _retrieve(user_prompt, k=6)
    context_block = "NO CONTEXT LOADED." if not ctx else "\n\n".join([f"[{src}] {text}" for (text, src, s) in ctx])

    augmented = (
        "You are producing a PRE-OP CHECKLIST for ER surgery PREP (SIMULATION ONLY).\n"
        "Use ONLY the REGISTRY CONTEXT below. If data is missing, say so clearly.\n\n"
        "REGISTRY CONTEXT:\n" + context_block + "\n\n"
        "USER QUESTION:\n" + user_prompt + "\n\n" + SCHEMA_ER_JSON
    )

    return client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=history + [{"role": "user", "parts": [{"text": augmented}]}],
        config=types.GenerateContentConfig(
            system_instruction=persona,
            response_mime_type="application/json",
            temperature=0.2,
            top_p=0.9,
        ),
    )


# ---------- Chat input ----------
prompt = st.chat_input("Type your message...")
if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build history in Gemini format (all but current user message)
    history = [{"role": m["role"], "parts": [{"text": m["content"]}]} for m in st.session_state.messages[:-1]]

    # Stream the model's response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        if mode == "ER Pre-Op Assistant (SIM)":
            stream = _run_er(history, prompt)
            for chunk in stream:
                if getattr(chunk, "text", None):
                    streamed_text += chunk.text
                    placeholder.markdown("_Generating structured checklist…_")
            # Try to render JSON nicely
            try:
                data = json.loads(streamed_text) if streamed_text else {}
            except json.JSONDecodeError:
                data = {}

            if data:
                st.subheader("Pre-Op Checklist (SIM)")
                st.write(f"**Patient match:** {data.get('patient_match', 'unknown')}")
                st.write(f"**Chief concern:** {data.get('chief_concern', '')}")
                st.write("---")
                st.write("### Checklist")
                for row in data.get("pre_op_checklist", []):
                    section = row.get("section", "Other")
                    item = row.get("item", "")
                    rationale = row.get("rationale", "")
                    evidence = ", ".join(row.get("evidence", []))
                    st.markdown(f"- **{section}** — {item}\n  - _Rationale:_ {rationale}\n  - _Evidence:_ {evidence}")

                if data.get("red_flags"):
                    st.error("**Red flags**")
                    for r in data["red_flags"]:
                        ev = ", ".join(r.get("evidence", []))
                        st.markdown(f"- {r.get('item','')} — {r.get('rationale','')}  \n  _Evidence:_ {ev}")

                if data.get("contraindications"):
                    st.warning("**Contraindications**")
                    for c in data["contraindications"]:
                        ev = ", ".join(c.get("evidence", []))
                        st.markdown(f"- {c.get('item','')} — {c.get('rationale','')}  \n  _Evidence:_ {ev}")

                if data.get("missing_information_requests"):
                    st.info("**Missing information requests**")
                    for q in data["missing_information_requests"]:
                        st.markdown(f"- {q}")

                st.caption(data.get("disclaimer", "This is a simulation and not medical advice."))
            else:
                placeholder.markdown(streamed_text or "_(no response)_")
        else:
            # General chat mode (stream as text)
            stream = _run_general(history, prompt)
            for chunk in stream:
                if getattr(chunk, "text", None):
                    streamed_text += chunk.text
                    placeholder.markdown(streamed_text)

        # Finalize assistant message in history
        st.session_state.messages.append(
            {"role": "model", "content": streamed_text or "_(no response)_"})
