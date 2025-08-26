# Knowledge Marketplace Enhancement: Semantic Data Ingestion & Search  

<code>
.
├── LICENSE
├── MS_Teams_Extension
│   ├── km_launcher.sh
│   ├── km_launcher.sh.visible_executions_in_terminal
│   ├── km_launcher.yml
│   └── README.md
├── README.md
├── structure.txt
└── WebApp_ChatBot
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
    ├── test_notebooks
    │   ├── TEST_OpenAI-API-Key.ipynb
    │   └── TEST_SharePoint_Connection.ipynb
    ├── tmp
    ├── uploads
    └── vectorstore
</code>

---

## 📌 Current Practice  
**Data Capture**  
- Users fill in details via a Power Apps form (link).  
- Submitted values are stored in the corresponding SharePoint List within Knowledge Marketplace (KM).  

**Data Search**  
- Users query a WebApp-based chatbot.  
- Relevant entries from KM may be returned.  

---

## ⚠️ Pain Points  
**Data Capture**  
- Lengthy: Users find it inconvenient to complete the 8-field Power Apps form.  
- Accessibility: Users must remember and access a specific link to submit new entries in KM SharePoint.  

**Data Search**  
- Performance: Chatbot suffers from slow loading during initial startup.  
- Reliability: Search often fails, returning zero results even for relevant queries.  

---

## 🚀 Proposed Solution  

### 1. Effortless Data Ingestion + Semantic Indexing  
**Effortless Ingestion**  
- Simplified user interaction: trigger via web application or Copilot Agent.  
- MS Teams Message Extension: Users can type `/km` directly in Teams.  
- Drag-and-drop file uploads into WebApp or Copilot Agent.  
- LLM-powered extraction: automatically parses information and uploads it into KM SharePoint List.  

**Semantic Indexing**  
- Convert textual entries into vector embeddings.  
- Store in a Vector Database alongside SharePoint storage.  

---

### 2. Semantic Search  
- Encode user queries into semantic vectors.  
- Compute similarity between query vector and KM records.  
- Return top-K most relevant results (e.g., top-2 or top-3).  

---

### 3. Unified Chatbot  
- Single chatbot interface that supports:  
  - **Data ingestion** (form-free, AI-assisted entry).  
  - **Semantic search** (reliable, context-aware retrieval).  

---

✅ With this approach, Knowledge Marketplace becomes more **accessible**, **reliable**, and **user-friendly** in terms of streamlining both data entry and discovery.  

