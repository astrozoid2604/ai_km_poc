# app.py
import os
import io
import re
import json
import time
import uuid
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# ========== Config ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.stop()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", ".chroma")
SHAREPOINT_ENABLED = os.getenv("SHAREPOINT_ENABLED", "false").lower() == "true"

SP_TENANT = os.getenv("SP_TENANT")
SP_SITE_URL = os.getenv("SP_SITE_URL")
SP_LIST_NAME = os.getenv("SP_LIST_NAME")
SP_CLIENT_ID = os.getenv("SP_CLIENT_ID")
SP_CLIENT_SECRET = os.getenv("SP_CLIENT_SECRET")

# ========== OpenAI ==========
from openai import OpenAI
oai = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-large"
GEN_MODEL = "gpt-4o-mini"

def embed_text(text: str) -> List[float]:
    text = text or ""
    resp = oai.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def llm_structured_extract(corpus: str, content_owner: str, function: str, site: str) -> Dict[str, str]:
    sys = (
        "You are an expert knowledge engineer. From the provided corpus, extract:\n"
        "1) Title: a concise representative title\n"
        "2) Content Summary: 1–2 cohesive paragraphs\n"
        "3) Benefits: 1–2 cohesive paragraphs highlighting business benefits\n"
        "Return a strict JSON object with keys: Title, ContentSummary, Benefits."
    )
    user = f"CORPUS:\n{corpus[:12000]}"  # keep prompt safe
    r = oai.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.2
    )
    text = r.choices[0].message.content
    # simple JSON parse attempt
    try:
        j = json.loads(text)
        return {
            "Title": j.get("Title","").strip(),
            "ContentSummary": j.get("ContentSummary","").strip(),
            "Benefits": j.get("Benefits","").strip(),
            "ContentOwner": content_owner,
            "Function": function,
            "Site": site,
        }
    except Exception:
        # fallback minimal
        return {
            "Title": "Untitled Knowledge Asset",
            "ContentSummary": text.strip()[:2000],
            "Benefits": "",
            "ContentOwner": content_owner,
            "Function": function,
            "Site": site,
        }

# ========== Chroma Vector DB ==========
import chromadb
try:
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
except TypeError:
    # compatibility with older chromadb API
    from chromadb.config import Settings
    client = chromadb.Client(Settings(persist_directory=VECTOR_DB_PATH))

collection = client.get_or_create_collection(
    name="ai4km_assets",
    metadata={"hnsw:space":"cosine"}
)

def upsert_asset_vector(record_id: str, text: str, metadata: Dict[str, Any], vector: List[float]):
    # Store exactly 1 vector per record to preserve 1:1 mapping.
    # If existing, delete then re-add (Chroma add() does not update).
    try:
        collection.delete(ids=[record_id])
    except Exception:
        pass
    collection.add(
        ids=[record_id],
        documents=[text],
        metadatas=[metadata],
        embeddings=[vector]
    )

def search_topk(query: str, k: int = 3):
    qvec = embed_text(query)
    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["ids","documents","metadatas","distances"]
    )
    # distances are cosine distance if using cosine space; convert to similarity = 1 - distance (approx)
    out = []
    for i in range(len(res.get("ids",[[]])[0])):
        out.append({
            "RecordId": res["ids"][0][i],
            "Document": res["documents"][0][i],
            "Metadata": res["metadatas"][0][i],
            "Score": float(1 - res["distances"][0][i]) if "distances" in res else None
        })
    return out

# ========== Local metadata store (CSV fallback) ==========
LOCAL_META_CSV = "vectorstore/metadata.csv"
os.makedirs("vectorstore", exist_ok=True)
if not os.path.exists(LOCAL_META_CSV):
    pd.DataFrame(columns=["RecordId","Title","ContentSummary","Benefits","ContentOwner","Function","Site"]).to_csv(LOCAL_META_CSV, index=False)

def local_meta_upsert(row: Dict[str,str]):
    df = pd.read_csv(LOCAL_META_CSV)
    df = df[df["RecordId"] != row["RecordId"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOCAL_META_CSV, index=False)

def local_meta_lookup(record_ids: List[str]) -> pd.DataFrame:
    df = pd.read_csv(LOCAL_META_CSV)
    return df[df["RecordId"].isin(record_ids)].copy()

# ========== SharePoint Client (when enabled) ==========
def sharepoint_available() -> bool:
    return SHAREPOINT_ENABLED and all([SP_SITE_URL, SP_LIST_NAME, SP_CLIENT_ID, SP_CLIENT_SECRET])

def sharepoint_add_item(row: Dict[str, str]) -> bool:
    from office365.sharepoint.client_context import ClientContext
    from office365.runtime.auth.client_credential import ClientCredential
    try:
        ctx = ClientContext(SP_SITE_URL).with_credentials(ClientCredential(SP_CLIENT_ID, SP_CLIENT_SECRET))
        sp_list = ctx.web.lists.get_by_title(SP_LIST_NAME)
        item_properties = {
            "Title": row["Title"],
            "ContentSummary": row["ContentSummary"],
            "Benefits": row["Benefits"],
            "ContentOwner": row["ContentOwner"],
            "Function": row["Function"],
            "Site": row["Site"],
            "RecordId": row["RecordId"]
        }
        item = sp_list.add_item(item_properties)
        ctx.execute_query()
        return True
    except Exception as e:
        st.warning(f"SharePoint write failed, storing locally instead. Details: {e}")
        return False

# ========== File parsers ==========
from PyPDF2 import PdfReader
from docx import Document
import tempfile

def read_pdf(file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp.flush()
        reader = PdfReader(tmp.name)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(texts)

def read_docx(file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file.read())
        tmp.flush()
        doc = Document(tmp.name)
        return "\n".join([p.text for p in doc.paragraphs])

def read_xlsx(file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(file.read())
        tmp.flush()
        df = pd.read_excel(tmp.name, sheet_name=None)
        parts = []
        for name, sdf in df.items():
            parts.append(f"## Sheet: {name}")
            parts.append(sdf.to_string(index=False))
        return "\n".join(parts)

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def consolidate_files(files) -> str:
    corpus_parts = []
    for f in files:
        name = f.name.lower()
        if name.endswith(".pdf"):
            corpus_parts.append(read_pdf(f))
        elif name.endswith(".docx"):
            corpus_parts.append(read_docx(f))
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            corpus_parts.append(read_xlsx(f))
        elif name.endswith(".txt"):
            corpus_parts.append(read_txt(f))
        else:
            corpus_parts.append(read_txt(f))  # naive fallback
    return "\n\n".join([p for p in corpus_parts if p.strip()])

# ========== Validators ==========
AMGEN_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@amgen\.com$")
SITE_RE = re.compile(r"^[A-Za-z]{3}$")

def to_title_case_words(s: str) -> str:
    return " ".join([w.capitalize() for w in re.split(r"\s+", (s or "").strip()) if w])

# ========== UI ==========
st.set_page_config(page_title="AI4KM - Local RAG", page_icon="💡", layout="wide")

if "mode" not in st.session_state:
    st.session_state.mode = None

st.title("AI4KM (AI for Knowledge Marketplace) - Local PoC")

col1, col2 = st.columns(2)
with col1:
    if st.button("Data Ingestion", use_container_width=True):
        st.session_state.mode = "ingest"
with col2:
    if st.button("Intelligent Search", use_container_width=True):
        st.session_state.mode = "search"

st.divider()

# ---------- MODE: INGEST ----------
if st.session_state.mode == "ingest":
    st.subheader("Data Ingestion Mode")

    with st.form("ingestion_form"):
        email = st.text_input("Amgen email address", placeholder="jdoe@amgen.com").strip().lower()
        site = st.text_input("Site (3-letter code)", placeholder="e.g., ASM").strip().upper()
        function = to_title_case_words(st.text_input("Function/Department (nice case)", placeholder="Digital Technology & Innovation"))
        files = st.file_uploader("Attach reference file(s)", accept_multiple_files=True,
                                 type=["pdf","docx","xlsx","xls","txt"])

        submitted = st.form_submit_button("Process")

    if submitted:
        # validations
        if not AMGEN_EMAIL_RE.match(email):
            st.error("Please enter a valid Amgen email (…@amgen.com).")
            st.stop()
        if not SITE_RE.match(site):
            st.error("Site must be a 3-letter code.")
            st.stop()
        if not function:
            st.error("Function/Department is required.")
            st.stop()
        if not files:
            st.error("Please attach at least one file.")
            st.stop()

        with st.spinner("Reading and consolidating files…"):
            corpus = consolidate_files(files)
            if not corpus.strip():
                st.error("No readable text found in the uploaded files.")
                st.stop()

        with st.spinner("Generating Title, Content Summary and Benefits via LLM…"):
            meta = llm_structured_extract(corpus, email, function, site)

        # Show preview
        st.success("Draft extracted. Please review:")
        st.write(f"**Title**: {meta['Title']}")
        st.write("**Content Summary**")
        st.write(meta["ContentSummary"])
        st.write("**Benefits**")
        st.write(meta["Benefits"])

        # Confirmation loop
        st.info("Do you want to submit a new knowledge asset?")
        coly, coln = st.columns(2)
        with coly:
            yes = st.button("Yes — Submit")
        with coln:
            no = st.button("No — Cancel")

        if yes:
            record_id = str(uuid.uuid4())

            with st.spinner("Creating embedding and saving to vector DB…"):
                # Build a single representative document string (truncate if extremely long)
                doc_for_vector = corpus
                if len(doc_for_vector) > 200000:   # simple guard
                    doc_for_vector = doc_for_vector[:200000]

                vec = embed_text(doc_for_vector)
                # 1:1 mapping: exactly one vector per record
                upsert_asset_vector(
                    record_id=record_id,
                    text=doc_for_vector,
                    metadata={
                        "Title": meta["Title"],
                        "ContentOwner": meta["ContentOwner"],
                        "Function": meta["Function"],
                        "Site": meta["Site"]
                    },
                    vector=vec
                )

            row = {
                "RecordId": record_id,
                **meta
            }

            saved = False
            if sharepoint_available():
                st.write("Attempting SharePoint write…")
                if sharepoint_add_item(row):
                    saved = True

            if not saved:
                local_meta_upsert(row)
                st.info("Saved locally (CSV). You can switch to SharePoint later by setting SHAREPOINT_ENABLED=true.")

            st.success(f"New knowledge asset submitted with RecordId: {record_id}")

            again = st.button("Ingest another?")
            if again:
                st.session_state.mode = "ingest"
                st.rerun()

        if no:
            st.session_state.mode = None
            st.rerun()

# ---------- MODE: SEARCH ----------
if st.session_state.mode == "search":
    st.subheader("Intelligent Search Mode")
    query = st.text_area("Please explain what knowledge asset or topic you are interested in")
    run = st.button("Search", type="primary")

    if run:
        if not query.strip():
            st.error("Please enter a query.")
            st.stop()

        with st.spinner("Searching similar knowledge assets…"):
            hits = search_topk(query, k=3)

        if not hits:
            st.warning("No matches found.")
        else:
            # Pull metadata rows either from SharePoint (not fetched here for simplicity) or local CSV
            # Since we stored full metadata in Chroma, we can display directly;
            # Also try to enrich from local CSV if present.
            rows = []
            for h in hits:
                m = h["Metadata"]
                rows.append({
                    "RecordId": h["RecordId"],
                    "Similarity": round(h["Score"], 4) if h["Score"] is not None else None,
                    "Title": m.get("Title",""),
                    "ContentOwner": m.get("ContentOwner",""),
                    "Function": m.get("Function",""),
                    "Site": m.get("Site","")
                })

            df = pd.DataFrame(rows)
            st.write("### Top Matches")
            st.dataframe(df, use_container_width=True)

            # Build an Excel with the matched records, including Content Summary + Benefits if available locally
            # Merge with local CSV (best-effort). If SharePoint-only, you can extend to fetch by RecordId.
            local_extra = local_meta_lookup([r["RecordId"] for r in rows])
            out = df.merge(local_extra, on="RecordId", how="left", suffixes=("",""))

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                out.to_excel(writer, index=False, sheet_name="Matches")
            st.download_button(
                "Download matches as Excel",
                data=buffer.getvalue(),
                file_name="ai4km_search_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Confirmation loop
        st.info("Do you want to do another search?")
        coly, coln = st.columns(2)
        with coly:
            again = st.button("Yes — Search again")
        with coln:
            exit_ = st.button("No — Exit")

        if again:
            st.session_state.mode = "search"
            st.rerun()
        if exit_:
            st.session_state.mode = None
            st.rerun()
