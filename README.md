# 📄 RAGdocQA: Intelligent Document Query System

**RAGdocQA** is an advanced Retrieval-Augmented Generation (RAG) system designed to interact with complex PDF documents using natural language. Unlike standard PDF chat tools, RAGdocQA is engineered to handle **unstructured and messy data**—such as raw engineering cutoff tables—and convert them into structured, actionable insights using Google's Gemini Pro AI.

---

## 🚀 Key Features & Differentiators

### 1. Intelligent Data Parsing (The "Messy Data" Solver)
Most RAG systems fail when extracting data from poorly formatted PDF tables (e.g., cutoff lists with broken lines like `-- --`). RAGdocQA uses specialized prompt engineering to:
*   Identify and reconstruct fragmented table rows.
*   Filter out noise characters.
*   **Predictively structure** raw text into clean Markdown tables.

### 2. Hybrid Persistence Architecture
*   **Permanent Knowledge Base:** Files uploaded to the system are **permanently ingested** into a local vector database. You build the knowledge base once, and it persists across restarts.
*   **Efficient Inference:** The application separates "Training" (database building) from "Inference" (chatting), ensuring zero latency on startup.

### 3. Transparent AI
*   **Source Citation:** Every answer includes a "View Source Documents" expandable section, allowing users to verify the exact text snippet used by the AI to generate the response.
*   **Robust Error Handling:** Built-in retry mechanisms with exponential backoff ensure stability effectively even under unstable cloud network conditions (handling 504 Timeouts).

---

## 🎯 Objectives

*   **Democratize Information Access:** Allow users to query complex technical documents (cutoffs, project reports, legal docs) without manually searching through hundreds of pages.
*   **Structure the Unstructured:** Convert raw, human-unreadable PDF dumps into clean, machine-readable tables and summaries.
*   **Seamless Deployment:** Provide a solution that works identically on a local developer machine and a cloud environment (Streamlit Cloud) without configuration changes.

---

## 🛠️ How It Works

1.  **Ingestion:**
    *   PDF files are placed in the `data/` directory or uploaded via the UI.
    *   The `pypdf` library extracts raw text.
    *   Text is split into semantic chunks (2000 characters) to optimize context retrieval.
2.  **Embedding:**
    *   Chunks are converted into vector embeddings using **Google Generative AI Embeddings (`models/text-embedding-004`)**.
    *   These vectors are stored in a persistent **ChromaDB** vector store (`chroma_db_final/`).
3.  **Retrieval & Generation:**
    *   User asks a question (e.g., *"What is the cutoff for CS in RV College?"*).
    *   The system performs a **Similarity Search** to find the most relevant chunks.
    *   **Google Gemini 1.5 Flash** acts as the reasoning engine to synthesize an answer from these chunks, applying strict formatting rules.

---

## 💡 Applications & Use Examples

### 🎓 Education & Counseling
*   **Scenario:** A student looking for college admission details.
*   **Query:** *"List the computer science cutoff ranks for standard general merit category in Mysore colleges."*
*   **Result:** A neat table comparing cutoffs for all matching colleges, extracted from a 400-page PDF.

### 🏢 Corporate Knowledge Management
*   **Scenario:** An employee needing details from old project reports.
*   **Query:** *"What were the specific objectives listed in the Career Path Forecasting project report?"*
*   **Result:** A bulleted list of objectives extracted strictly from the "Purpose" section of the document.

### ⚖️ Legal & Compliance
*   **Scenario:** extracting clauses from a messy scanned contract.
*   **Query:** *"What are the termination conditions mentions?"*
*   **Result:** A summary of clauses without the OCR noise.

---

## 💻 Tech Stack

*   **LLM:** Google Gemini 1.5 Flash (via `langchain-google-genai`)
*   **Vector Store:** ChromaDB (Persistent)
*   **Framework:** LangChain & Streamlit
*   **Data Processing:** PyPDF
*   **Language:** Python 3.10+

---

## 🏃‍♂️ Getting Started

### Prerequisites
*   Python 3.10+
*   Google API Key

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/RAGdocQA.git
    cd RAGdocQA
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

### Running the App

1.  **Build the Database (First Run / New Files)**
    If you have files in `data/`, run this once:
    ```bash
    python build_db.py
    ```

2.  **Launch the Chatbot**
    ```bash
    python -m streamlit run app.py
    ```
