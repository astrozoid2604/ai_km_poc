<code>
.
├── ai4km.yaml
├── app.py
├── README.md
├── sharepoint_certificates
│   ├── ai4km_cert.cer
│   ├── ai4km_cert.key
│   ├── ai4km_cert.pfx
│   ├── cert.pem
│   ├── cert.zip
│   └── privkey.pem
├── structure.txt
├── test_notebooks
│   ├── TEST_OpenAI-API-Key.ipynb
│   └── TEST_SharePoint_Connection.ipynb
├── tmp
├── uploads
└── vectorstore
</code>


This guide explains how to configure your macOS environment to deploy RAG chatbot locally for the purpose of implementing Data Ingestion mode and Intelligent Search mode for Knowledge Marketplace.

[YouTube Video LINK](https://youtu.be/Q1tb7dmULoA)

[![YouTube Thumbnail](https://img.youtube.com/vi/Q1tb7dmULoA/hqdefault.jpg)](https://youtu.be/Q1tb7dmULoA)

---

## 0. Prerequisites (once)
- OS: macOS/Windows/Linux
- Conda: Miniconda/Anaconda installed
- Python: 3.10+
- OpenAI API key: Create and copy your key (`OPENAI_API_KEY`)
- (Optional) SharePoint: App registration with client credentials (see Section 4 below)

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
OPENAI_API_KEY=sk-REPLACE_ME

# Vector DB (Chroma) persistence directory
VECTOR_DB_PATH=.chroma

# Toggle SharePoint integration
SHAREPOINT_ENABLED=true

# ----- Only needed if SHAREPOINT_ENABLED=true -----
# --- The instructions on how to get all 8 SharePoint-related env variables below would be elaborated in Section 5 `SharePoint App Permissions` ---

# Tenant
SP_TENANT=yourtenant.onmicrosoft.com
SP_TENANT_ID=REPLACE_ME

# Target site & list
SP_SITE_URL=https://yourtenant.sharepoint.com/sites/ai4km
SP_LIST_NAME=AI4KM Knowledge Assets

#App credentials
SP_CLIENT_ID=REPLACE_ME
SP_CLIENT_SECRET=REPLACE_ME

#App certificates (P.S. SP_CERT_PEM_PATH below is fixed)
SP_CERT_THUMBPRINT=REPLACE_ME
SP_CERT_PEM_PATH=./sharepoint_certificates/privkey.pem
```

---

## 4. SharePoint App Permissions (optional, admin required)

SharePoint and Azure App registration are both only available in work/school O365 account and will not be available for `Microsoft Office 365 Personal` subcription for individuals/families. Therefore, the assumption here is that you need to create a `Microsoft 365 Business Basic` under free trial version of 30 days (extendable). Kindly follow the steps below sequentially.

### 4a. Create `Microsoft 365 Business Basic` account for 1 person in the company.
1. Search in search engine `Microsoft 365 Business Basic`. At the time of writing on 26 AUG 2025, the cost is USD$7.85 (tax included) per month. However, for this development purpose, can opt for **Try free for one month**.
2. You will be navigated to complete **Subcription & account details**. Can fill this section up accordingly.
3. Next, you will be redirected to complete **Sign-in details**. Please note that `Domain name` here is none other than `SP_TENANT` env variable.
4. Next, you will be redirected to **Add payment, confirm & complete order**. Ensure the **Sold-to address** is the same as registered mailing address of your debit card/credit card. Then, click **Start Trial**.

### 4b. Prepare the SharePoint List
1. If you plan to use **SharePoint Online** (i.e. `SHAREPOINT_ENABLED=true`) for storing metadata instead of the lcoal CSV fallback (i.e. `SHAREPOINT_ENABLED=false`), you will need to prepare a SharePoint List.
2. Go to https://<your_tenant>.sharepoint.com, and click the SharePoint start page (i.e. home icon).
3. Create a new site with exact name `ai4km`
4. Within this `ai4km` site, create a new list with exact name `AI4KM Knowledge Assets`.
5. Upon doing these, you will have 2 environment variables, namely, `SP_SITE_URL=https://<your_tenant>.sharepoint.com/sites/ai4k`, and `SP_LIST_NAME=AI4KM Knowledge Assets`.
6. Add these following columns

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

### 4c. Register new app in Azure portal
1. Go to https://portal.azure.com.
2. Search for **App Registrations**
3. Click **+ New registration**. Fill up the **Name** as `ai4km-sp`. For the **Supported account types**, choose `Accounts in this organizational directory only (only - Single tenant). Then, click **Register**`.
4. If you refresh the **App registrations**, you will see app with **Display Name** equals to `ai4km-sp` being created.
5. If you click into the `ai4km-sp` app, you will be redirected to **Overview** selection in left panel.
6. In this **Overview** selection, `SP_CLIENT_ID` env variable is equal to **Application (client) ID** and `SP_TENANT_ID` is equal to **Directory (tenant) ID**.
7. In the left panel, you will see **Certificates & secrets**. Click on the **Client secrets** tab and click **+ New client secret**. Please note that `SP_CLIENT_SECRET` is equal to **Value** and that you can only see the full string of **Value** for the first time you visit this page. If you were to refresh or navigate away and back, **Value** would be hidden and you need to create a new client secret.
8. In the left panel, click **API permissions**. Click **+ Add a permission**. Under **Microsoft API**, select **SharePoint** > **Application permission**. Then, under **Sites**, select `Sites.FullControl.All` and `Sites.ReadWrite.All`. Then, click **Add permissions**. Next, click on **Grant admin consent for** > **Yes**.
9. In your command terminal, go to `./WebApp_ChatBot/sharepoint_certificates` and execute the following commands.

```bash
# creates ai4km_cert.pfx and ai4km_cert.cer (public) with no export password
openssl req -x509 -nodes -days 730 -newkey rsa:2048 -keyout ai4km_cert.key -out ai4km_cert.cer -subj "/CN=ai4km-app"

# package to PFX (set a password if you like)
openssl pkcs12 -export -out ai4km_cert.pfx -inkey ai4km_cert.key -in ai4km_cert.cer -passout pass:

# Extract private key (no password)
openssl pkcs12 -in ai4km_cert.pfx -nocerts -nodes -out privkey.pem

# Extract public cert
openssl pkcs12 -in ai4km_cert.pfx -clcerts -nokeys -out cert.pem

# Extract private key PEM (no password prompt if your PFX has none)
openssl pkcs12 -in ai4km_cert.pfx -nocerts -nodes -out privkey.pem
```

10. `SP_CERT_THUMBPRINT` env variable is equal to the output of below command.

```bash
# Compute SHA1 thumbprint (remove colons, use uppercase)
openssl x509 -in cert.pem -noout -fingerprint -sha1 | awk -F= '{print $2}' | tr -d ':' | tr '[:lower:]' '[:upper:]'
```

11. Go back to Azure portal at https://portal.azure.com. Go to **App Registrations** > **ai4km-sp** > **Certificates & secrets**. Click **Upload certificate** and upload `./WebApp_ChatBot/sharepoint_certificates/ai4km_cert.cer`.

---

## 5. Create the Streamlit App (Single File)

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

## 6. Run the App

```bash 
conda activate ai4km
streamlit run app.py
```

Your browser opens to the Streamlit interface with **two big buttons**:
- **Data Ingestion**: Runs the full ingestion workflow (metadata prompts, file upload, LLM extraction, vector upsert, SharePoint/local save) with a confirmation loop.
- **Intelligent Search**: Query --> **Top-3 Cosine Similarity matches** --> inline table + **Excel Download** + confirmation loop.

---

## 7. How the 1:1 Mapping is Guaranteed

- For each new asset, the app creates a unique `RecordId` (UUID).
- Exactly **one embedding** is computed from the **consolidated corpus** per asset and stored in Chroma with **that same** `RecordId`.
- The same `RecordId` is stored in the **SharePoint List** (or local CSV).
- This preserves a strict **1 asset** <--> **1 vector** mapping.

---

## 8. Switching to SharePoint Later

1. Set `SHAREPOINT_ENABLED=true` in `.env` and fill all SP_* variables.
2. Ensure your list exists and internal names match.
3. Re-run `streamlit run app.py`.
4. New ingestions write to SharePoint, existing local rows remain in CSV (you can build a one-time migration script if needed)

---

## 9. Testing Checklist

- Ingestion with **PDF + DOCS + XLSX** together
- Email validation only accept `@amgen.com`
- `ASM` acceptep; `ASMN` rejected (only 3-character string)
- Function normalized to **Title Case**
- Search returns **top-3** and Excel downloads with the fields
- Toggle SharePoint on/off and verify behaviour

---

## 10. Troubleshooting

- `ModuleNotFoundError`: Confirm the conda env is active and packages installed.
- **Chroma errors**: Delete `.chroma` directory and re-run (it will rebuild).
- **SharePoint permission**: Confirm the app has site access and list internal names match the code.
- **OpenAI quota/auth**: Check `OPENAI_API_KEY`, and try a very small test file to verify flow.
