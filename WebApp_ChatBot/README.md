## 0. Prerequisites (once)
- OS: macOS/Windows/Linux
- Conda: Miniconda/Anaconda installed
- Python: 3.10+
- OpenAI API key: Create and copy your key (`OPENAI_API_KEY`)
- (Optional) SharePoint: App registration with client credentials (see Section 5 below)

---

## 1. Create the project

```bash
# Create folders for persistence
mkdir -p .chroma vectorstore tmp uploads
```

Project directories
- `.chroma/` – local ChromaDB index (auto-created, not versioned)  
- `vectorstore/` – stores metadata CSV (created automatically); repo ships empty  
- `tmp/` – runtime scratch space  
- `uploads/` – temporary area for file uploads

> These folders are **git-ignored** to avoid committing large/generated or sensitive data. When you clone, just run the app once and the folders will be created automatically.

---

## 2. Create the conda environment & install packages

```bash
conda env create -f ai4km.yaml
```

---

## 3. Add your environment variables

Create a file named `.env` in the project root:

```bash
# ----- Required -----
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Vector DB (Chroma) persistence directory
VECTOR_DB_PATH=.chroma

# Toggle SharePoint integration (true|false). If false, metadata is stored locally.
SHAREPOINT_ENABLED=false

# ----- Only needed if SHAREPOINT_ENABLED=true -----
SP_TENANT=yourtenant.onmicrosoft.com
SP_SITE_URL=https://yourtenant.sharepoint.com/sites/YourSiteName
SP_LIST_NAME=AI4KM Knowledge Assets
SP_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
SP_CLIENT_SECRET=your-client-secret
```

---

## 4. Prepare the SharePoint List (only if using SharePoint now)

If you plan to use **SharePoint Online** for storing metadata instead of the local CSV fallback, you will need to prepare a SharePoint List.

### 4a. Create a New SharePoint List

1.  Go to your SharePoint Site (e.g.,
    `https://<tenant>.sharepoint.com/sites/ai4km`).
2.  Click on **Settings (⚙️)** → **Site contents**.
3.  Select **+ New → List**.
4.  Choose **Blank list**, and name it:

```{=html}
<!-- -->
```
    AI4KM Knowledge Assets

### 4b. Add the Required Columns

If `SHAREPOINT_ENABLED=true`, add these columns (Display Name --> Type --> Internal Name you must ensure). In the newly created list, add the following columns (use the exact internal names):

 | Display Name    |Type                    |Internal Name    |
 | ----------------|------------------------|-----------------|
 | Title           |Single line of text     |Title (default)  |
 | ContentSummary  |Multiple lines of text  |ContentSummary   |
 | Benefits        |Multiple lines of text  |Benefits         |
 | ContentOwner    |Single line of text     |ContentOwner     |
 | Function        |Single line of text     |Function         |
 | Site            |Single line of text     |Site             |
 | RecordId        |Single line of text     |RecordId         |

> The app writes to those internal names. If your tenant forces encoded names, adjust in the code (search "SharePoint field map").
> The **List name** must be exactly: `AI4KM Knowledge Assets`
> Internal names are case-sensitive and must match the table above.
> If you enable SharePoint integration later (`SHAREPOINT_ENABLED=true` in `.env`), the chatbot will write to this list.
> If SharePoint is nott enabled, the project falls back to a lcoal CSV in `vectorstore/metadata.csv`.

---

## 5. SharePoint App Permissions (optional, admin required)

If you want immediate PoC without admin friction, keep `SHAREPOINT_ENABLED=false` now (local CSV).

To enable SharePoint later:
1. Register an Azure AD app (Entra ID) --> get **Client ID** + **Client Secret**
2. Grant **Sites.Selected** (Graph) or **AllSites.FullControl** (SharePoint) app-only permissions and consent.
3. If using **Sites.Selected**, assign the app access to your site (via Graph or SharePoint admin center).
4. Put credentials into `.env`, set `SHAREPOINT_ENABLED=true`.

---

## 6. Create the Streamlit App (Single File)

Create `app.py` in the project root with the code below.

This implements:
- Two modes: Data Ingestion & Intelligent Search
- Validations (Amgen email, 3‑char site, nice‑case function)
- Multi‑file upload (PDF/DOCX/XLSX/TXT) → consolidated corpus
- LLM extraction of Title/Summary/Benefits
- 1:1 vector per knowledge asset (document‑level embedding) stored in Chroma
- SharePoint write (or local CSV fallback)
- Search with cosine similarity (top‑3), inline results + Excel download
- Confirmation loops for both modes

--- 

## 7. Run the App

```bash 
conda activate ai4km
streamlit run app.py
```

Your browser opens to the Streamlit interface with **two big buttons**:
- **Data Ingestion**: Runs the full ingestion workflow (metadata prompts, file upload, LLM extraction, vector upsert, SharePoint/local save) with a confirmation loop.
- **Intelligent Search**: Query --> **Top-3 Cosine Similarity matches** --> inline table + **Excel Download** + confirmation loop.

---

## 8. How the 1:1 Mapping is Guaranteed

- For each new asset, the app creates a unique `RecordId` (UUID).
- Exactly **one embedding** is computed from the **consolidated corpus** per asset and stored in Chroma with **that same** `RecordId`.
- The same `RecordId` is stored in the **SharePoint List** (or local CSV).
- This preserves a strict **1 asset** <--> **1 vector** mapping.

---

## 9. Switching to SharePoint Later

1. Set `SHAREPOINT_ENABLED=true` in `.env` and fill all SP_* variables.
2. Ensure your list exists and internal names match.
3. Re-run `streamlit run app.py`.
4. New ingestions write to SharePoint, existing local rows remain in CSV (you can build a one-time migration script if needed)

---

## 10. Testing Checklist

- Ingestion with **PDF + DOCS + XLSX** together
- Email validation only accept `@amgen.com`
- `ASM` acceptep; `ASMN` rejected (only 3-character string)
- Function normalized to **Title Case**
- Search returns **top-3** and Excel downloads with the fields
- Toggle SharePoint on/off and verify behaviour

---

## 11. Troubleshooting

- `ModuleNotFoundError`: Confirm the conda env is active and packages installed.
- **Chroma errors**: Delete `.chroma` directory and re-run (it will rebuild).
- **SharePoint permission**: Confirm the app has site access and list internal names match the code.
- **OpenAI quota/auth**: Check `OPENAI_API_KEY`, and try a very small test file to verify flow.
