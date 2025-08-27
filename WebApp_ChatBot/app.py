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
os.chdir('/Users/jameslim/Desktop/GITHUB/ai_km_poc/WebApp_ChatBot')

# ========== Config ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.stop()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", ".chroma")
SHAREPOINT_ENABLED = os.getenv("SHAREPOINT_ENABLED", "false").lower() == "true"

# SharePoint (certificate-based app-only)
SP_TENANT      = os.getenv("SP_TENANT")             # e.g., yourtenant.onmicrosoft.com OR tenant GUID
SP_SITE_URL    = os.getenv("SP_SITE_URL")           # e.g., https://<tenant>.sharepoint.com/sites/ai4km
SP_LIST_NAME   = os.getenv("SP_LIST_NAME")          # e.g., AI4KM Knowledge Assets
SP_CLIENT_ID   = os.getenv("SP_CLIENT_ID")          # App (client) ID
SP_CERT_THUMB  = os.getenv("SP_CERT_THUMBPRINT")    # UPPERCASE hex, no colons
SP_CERT_PEM    = os.getenv("SP_CERT_PEM_PATH")      # path to PEM **private key** (e.g., ./privkey.pem)

# ========== OpenAI ==========
from openai import OpenAI
oai = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-large"
GEN_MODEL = "gpt-4o-mini"

def embed_text(text: str) -> List[float]:
    text = text or ""
    resp = oai.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def _coerce_json(s: str) -> dict:
    """Make a best-effort to parse JSON returned by an LLM."""
    if not isinstance(s, str):
        raise ValueError("Expected string")
    s = s.strip()

    # Strip Markdown code fences
    if s.startswith("```"):
        # remove first fence line
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl+1:]
        if s.endswith("```"):
            s = s[: -3].strip()

    # Keep only the outermost JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]

    # Normalize smart quotes/apostrophes to ASCII
    s = (s
         .replace("\u201c", '"').replace("\u201d", '"')  # “ ”
         .replace("\u2018", "'").replace("\u2019", "'")) # ‘ ’

    # Remove trailing commas before a closing brace/bracket
    import re
    s = re.sub(r",\s*(\}|\])", r"\1", s)

    return json.loads(s)

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
        j = _coerce_json(text)
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

    # Do NOT include "ids" here; Chroma returns ids regardless.
    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"]  # valid keys only
    )

    ids         = (res.get("ids") or [[]])[0]
    documents   = (res.get("documents") or [[]])[0]
    metadatas   = (res.get("metadatas") or [[]])[0]
    distances   = (res.get("distances") or [[]])[0]

    out = []
    for i in range(len(ids)):
        dist = float(distances[i]) if i < len(distances) and distances[i] is not None else None
        sim  = (1.0 - dist) if dist is not None else None  # cosine similarity approx
        out.append({
            "RecordId": ids[i],
            "Document": documents[i] if i < len(documents) else None,
            "Metadata": metadatas[i] if i < len(metadatas) else {},
            "Score": sim
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

# ========== SharePoint Client (certificate app-only) ==========
def sharepoint_available() -> bool:
    # All must be present to enable SharePoint path
    needed = [SP_SITE_URL, SP_LIST_NAME, SP_CLIENT_ID, SP_TENANT, SP_CERT_THUMB, SP_CERT_PEM]
    return SHAREPOINT_ENABLED and all(needed)

def _get_sp_ctx():
    """Return a ClientContext authenticated via client certificate."""
    from office365.sharepoint.client_context import ClientContext
    return ClientContext(SP_SITE_URL).with_client_certificate(
        tenant=SP_TENANT,
        client_id=SP_CLIENT_ID,
        thumbprint=SP_CERT_THUMB,
        cert_path=SP_CERT_PEM
    )

def sharepoint_add_item(row: Dict[str, str]) -> bool:
    """
    Write one item into the AI4KM Knowledge Assets list.
    Expects keys: Title, ContentSummary, Benefits, ContentOwner, Function, Site, RecordId
    """
    try:
        ctx = _get_sp_ctx()
        sp_list = ctx.web.lists.get_by_title(SP_LIST_NAME)
        item_properties = {
            "Title":          row["Title"],
            "ContentSummary": row["ContentSummary"],
            "Benefits":       row["Benefits"],
            "ContentOwner":   row["ContentOwner"],
            "Function":       row["Function"],
            "Site":           row["Site"],
            "RecordId":       row["RecordId"],
        }
        item = sp_list.add_item(item_properties)
        ctx.execute_query()  # commit
        return True
    except Exception as e:
        st.warning(f"SharePoint write failed, storing locally instead. Details: {e}")
        return False

def sharepoint_fetch_by_record_id(record_id: str) -> Dict[str, Any] | None:
    """Fetch one item by RecordId using OData $filter (no CAML)."""
    try:
        ctx = _get_sp_ctx()
        sp_list = ctx.web.lists.get_by_title(SP_LIST_NAME)

        items = (
            sp_list.items
            .select(["Id","Title","RecordId","ContentSummary","Benefits","ContentOwner","Function","Site"])  # ← list, not string
            .filter(f"RecordId eq '{record_id}'")
            .top(1)
        )
        ctx.load(items)
        ctx.execute_query()

        if len(items) == 0:
            return None

        it = items[0]
        p = it.properties
        return {
            "RecordId":       p.get("RecordId"),
            "Title":          p.get("Title"),
            "ContentSummary": p.get("ContentSummary"),
            "Benefits":       p.get("Benefits"),
            "ContentOwner":   p.get("ContentOwner"),
            "Function":       p.get("Function"),
            "Site":           p.get("Site"),
            "ItemId":         p.get("Id") or p.get("ID"),
        }
    except Exception as e:
        st.warning(f"SharePoint read failed; falling back to local metadata where possible. Details: {e}")
        return None

def sharepoint_lookup_record_ids(record_ids: List[str]) -> pd.DataFrame:
    """Fetch a small list of records by RecordId (one-by-one)."""
    out = []
    for rid in record_ids:
        row = sharepoint_fetch_by_record_id(rid)
        if row:
            out.append(row)
    return pd.DataFrame(out) if out else pd.DataFrame(
        columns=["RecordId","Title","ContentSummary","Benefits","ContentOwner","Function","Site","ItemId"]
    )

def generate_answer(query: str, docs: List[Dict[str, Any]]) -> str:
    """
    Use LLM to synthesize an answer from top-k retrieved documents.
    Each doc dict should have 'Title', 'ContentSummary', 'Benefits'.
    """
    if not docs:
        return "No relevant documents found to answer your question."

    context_parts = []
    for i, d in enumerate(docs, start=1):
        context_parts.append(
            f"[Doc {i}] Title: {d.get('Title','')}\n"
            f"Summary: {d.get('ContentSummary','')}\n"
            f"Benefits: {d.get('Benefits','')}\n"
        )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are an expert assistant that answers user questions based on provided enterprise knowledge assets. "
        "Use the context faithfully. If the context does not contain the answer, say you could not find it. "
        "Do not invent details."
    )
    user_prompt = f"User query:\n{query}\n\nContext:\n{context}\n\nAnswer the query in 1-2 paragraphs."

    r = oai.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.3
    )
    return r.choices[0].message.content.strip()

# ========== File parsers ==========
from PyPDF2 import PdfReader
from docx import Document
import tempfile
from pptx import Presentation


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

def read_pptx(file) -> str:
    """
    Extracts readable text from a .pptx:
      - slide body text (text frames)
      - table cell text
      - speaker notes (if present)
    Returns a single consolidated string.
    """
    import tempfile

    # Save uploaded file to a temp path for python-pptx
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
        tmp.write(file.read())
        tmp.flush()
        path = tmp.name

    prs = Presentation(path)
    parts = []

    for idx, slide in enumerate(prs.slides, start=1):
        parts.append(f"## Slide {idx}")

        # 1) Text frames on shapes
        for shape in slide.shapes:
            # Text frames (titles, content placeholders, text boxes)
            if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                tf = shape.text_frame
                for para in tf.paragraphs:
                    # Joins all runs in paragraph to preserve inline fragments
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        parts.append(text)

            # 2) Tables
            if hasattr(shape, "has_table") and shape.has_table:
                tbl = shape.table
                for row in tbl.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    row_text = " | ".join([c for c in cells if c])
                    if row_text:
                        parts.append(row_text)

        # 3) Speaker notes (if any)
        try:
            notes = slide.notes_slide.notes_text_frame.text
            if notes and notes.strip():
                parts.append("### Notes")
                parts.append(notes.strip())
        except Exception:
            # No notes slide attached or structure differs; ignore
            pass

    # Final consolidated text
    return "\n".join([p for p in parts if p.strip()])

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
        elif name.endswith(".pptx"):
            corpus_parts.append(read_pptx(f))
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

def _reset_ingest_state():
    st.session_state.pop("ing_meta", None)
    st.session_state.pop("ing_corpus", None)
    st.session_state.pop("ing_ready", None)
    st.session_state.pop("ing_last_record_id", None)

# init keys once
for k in ["ing_meta", "ing_corpus", "ing_ready", "ing_last_record_id"]:
    if k not in st.session_state:
        st.session_state[k] = None

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

    # Show the form only if we are not in the "confirm" step
    if not st.session_state.ing_ready:
        with st.form("ingestion_form"):
            email = st.text_input("Amgen email address", placeholder="jdoe@amgen.com").strip().lower()
            site = st.text_input("Site (3-letter code)", placeholder="e.g., ASM").strip().upper()
            function = to_title_case_words(
                st.text_input("Function/Department (nice case)", placeholder="Digital Technology & Innovation")
            )
            files = st.file_uploader(
                "Attach reference file(s)", accept_multiple_files=True,
                type=["pdf","docx","xlsx","xls","txt","pptx"]
            )
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

            # Persist results for the confirmation step on the next rerun
            st.session_state.ing_corpus = corpus
            st.session_state.ing_meta = meta
            st.session_state.ing_ready = True
            st.rerun()

    # Confirmation step (survives reruns)
    else:
        meta = st.session_state.ing_meta or {}
        st.success("Draft extracted. Please review:")
        st.write(f"**Title**: {meta.get('Title','')}")
        st.write("**Content Summary**")
        st.write(meta.get("ContentSummary",""))
        st.write("**Benefits**")
        st.write(meta.get("Benefits",""))

        st.info("Do you want to submit a new knowledge asset?")
        coly, coln = st.columns(2)
        yes = coly.button("Yes — Submit", key="yes_submit")
        no  = coln.button("No — Cancel", key="no_cancel")

        if yes:
            record_id = str(uuid.uuid4())
            corpus = st.session_state.ing_corpus or ""
            with st.spinner("Creating embedding and saving to vector DB…"):
                doc_for_vector = corpus[:200000] if len(corpus) > 200000 else corpus
                vec = embed_text(doc_for_vector)

                # 1:1 mapping
                upsert_asset_vector(
                    record_id=record_id,
                    text=doc_for_vector,
                    metadata={
                        "Title": meta.get("Title",""),
                        "ContentOwner": meta.get("ContentOwner",""),
                        "Function": meta.get("Function",""),
                        "Site": meta.get("Site","")
                    },
                    vector=vec
                )

            row = {"RecordId": record_id, **meta}

            saved = False
            if sharepoint_available():
                st.write("Attempting SharePoint write…")
                if sharepoint_add_item(row):
                    saved = True

            if not saved:
                local_meta_upsert(row)
                st.info("Saved locally (CSV). You can switch to SharePoint later by setting SHAREPOINT_ENABLED=true.")

            st.session_state.ing_last_record_id = record_id
            _reset_ingest_state()  # clear confirm state
            st.success(f"New knowledge asset submitted with RecordId: {record_id}")

            # Offer to ingest another
            if st.button("Ingest another?", key="ingest_again"):
                st.session_state.mode = "ingest"
                st.rerun()

        if no:
            _reset_ingest_state()
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

            # Build an Excel with the matched records, including Content Summary + Benefits
            rec_ids = [r["RecordId"] for r in rows]
            
            if sharepoint_available():
                sp_extra = sharepoint_lookup_record_ids(rec_ids)
            
                # rename overlapping columns so pandas doesn't complain
                sp_extra = sp_extra.rename(columns={
                    "Title": "Title_sp",
                    "ContentOwner": "ContentOwner_sp",
                    "Function": "Function_sp",
                    "Site": "Site_sp"
                })
            
                # bring only the fields we actually need from SP
                sp_cols = ["RecordId", "ContentSummary", "Benefits", "Title_sp", "ContentOwner_sp", "Function_sp", "Site_sp"]
                sp_extra = sp_extra[sp_cols] if not sp_extra.empty else sp_extra
            
                # left-merge preserves ranking order from df
                out = df.merge(sp_extra, on="RecordId", how="left")
            
                # optional: if your df lacked some fields, backfill from SP
                # out["Title"] = out["Title"].where(out["Title"].notna() & (out["Title"] != ""), out["Title_sp"])
                # out["ContentOwner"] = out["ContentOwner"].fillna(out["ContentOwner_sp"])
                # out["Function"] = out["Function"].fillna(out["Function_sp"])
                # out["Site"] = out["Site"].fillna(out["Site_sp"])
            
            else:
                local_extra = local_meta_lookup(rec_ids)
                out = df.merge(local_extra, on="RecordId", how="left", suffixes=("",""))

            # Right after merging into `out`
            doc_contexts = out.to_dict(orient="records")
            
            with st.spinner("Generating synthesized answer..."):
                answer = generate_answer(query, doc_contexts)
            
            st.write("### Synthesized Answer")
            st.write(answer)

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
