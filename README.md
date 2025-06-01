# MediQuery AI 🩺🤖

**MediQuery AI** is an advanced, AI-powered health information assistant designed to provide users with quick access to medical knowledge. It allows users to explore symptoms, ask health-related questions, and generate concise PDF reports. This project showcases a sophisticated backend built with Python and FastAPI, leveraging state-of-the-art Natural Language Processing (NLP) models and a hybrid search system to deliver relevant information from MedlinePlus data.

## ✨ Core Features

- **⚕️ Intelligent Health Q&A:** Ask complex health-related questions in natural language. Receive answers sourced from processed MedlinePlus data, powered by an advanced Retrieval Augmented Generation (RAG) pipeline.
- **🔎 Symptom Explorer:** Describe symptoms, and the AI identifies key categories using Zero-Shot Classification. It then fetches and presents relevant information for each identified category from its knowledge base.
- **📄 PDF Report Generation:** Obtain a structured PDF report summarizing your Q&A session or symptom exploration, complete with sources for easy reference.
- **🗣️ Multimodal Input (Backend Ready, Frontend Implemented for Text & Audio):**
  - **Text Input:** Core functionality for questions and symptom descriptions.
  - **Audio Input:** Speech-to-text transcription integrated into the frontend using browser APIs and a backend Whisper endpoint.
  - **Image Input (OCR):** Optical Character Recognition using EasyOCR on the backend, with frontend integration allowing users to upload images (e.g., medication labels) and use extracted text in their queries.
- **🧠 Advanced AI/NLP Pipeline:**
  - **Hybrid Search (RRF):** Combines FAISS vector search (semantic with `msmarco-distilbert-base-tas-b`) and BM25 keyword search (lexical) using Reciprocal Rank Fusion for optimized information retrieval accuracy.
  - **Extractive Question Answering:** Utilizes BioBERT (`dmis-lab/biobert-base-cased-v1.1-squad`) to accurately extract answers from the retrieved contextual information.
  - **Zero-Shot Text Classification:** Employs BART-Large-MNLI (`facebook/bart-large-mnli`) to intelligently categorize user-described symptoms for the Symptom Explorer.
  - **Text Summarization:** Includes a dedicated endpoint for text summarization using DistilBART (`sshleifer/distilbart-cnn-12-6`).

## 🛠️ Tech Stack

**Backend:**

- **Framework:** FastAPI (Python 3.9+)
- **NLP Models & Libraries:**
  - Hugging Face `transformers`, `sentence-transformers`
  - OpenAI `whisper` (Speech-to-Text)
  - `easyocr` (OCR)
  - `rank_bm25` (BM25 Keyword Search)
  - `faiss-cpu` (FAISS Vector Similarity Search)
  - **Models Used:**
    - `dmis-lab/biobert-base-cased-v1.1-squad` (Extractive QA)
    - `sentence-transformers/msmarco-distilbert-base-tas-b` (Embeddings for RAG)
    - `facebook/bart-large-mnli` (Zero-Shot Symptom Classification)
    - `sshleifer/distilbart-cnn-12-6` (Summarization)
- **PDF Generation:** `reportlab`
- **Web Server:** Uvicorn
- **Primary Data Source:** MedlinePlus (publicly available XML data for health topics)

**Frontend:**

- **Framework:** React with TypeScript (bootstrapped with Vite)
- **Styling:** Tailwind CSS
- **API Communication:** `fetch` API
- **Audio Recording:** Browser `MediaRecorder` API

## ⚙️ Project Structure

```
mediquery_ai/
├── backend/
│ ├── data/
│ │ └── medlineplus/
│ │ ├── mplus_topics_YYYY-MM-DD.xml (Example raw MedlinePlus XML)
│ │ ├── processed_health_topics.json (Cleaned, chunked data)
│ │ ├── health_topics.index (FAISS vector index)
│ │ └── chunk_metadata.json (Not strictly used if full_processed_data loaded by main.py)
│ ├── venv/ (Python virtual environment - add to .gitignore)
│ ├── data_preprocessor.py (Parses XML, chunks text using sentence tokenization)
│ ├── build_vector_store.py (Generates embeddings, builds FAISS index)
│ ├── main.py (FastAPI app: endpoints, model loading, RAG, BM25 in-memory)
│ └── requirements.txt (Python dependencies)
└── frontend/ (React application)
├── src/
│ ├── components/ (React components)
│ ├── App.tsx (Main app logic)
│ └── main.tsx (Entry point)
├── public/
├── .env (For VITE_API_URL)
└── package.json
```

## 🚀 Local Setup & Running

**Prerequisites:**

- Python (3.9 or higher recommended)
- Node.js (LTS version recommended) and Yarn (or npm)
- FFmpeg: Essential for `openai-whisper` audio processing. Download from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure the `bin` directory containing `ffmpeg.exe` is added to your system's PATH.

**Backend Setup:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abhinavfaldu082/Medi_Query.git
    cd medi_query/backend
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download MedlinePlus Data:**
    - Visit [MedlinePlus XML Health Topics](https://medlineplus.gov/xml.html).
    - Download the "Health Topics in XML" zip file (e.g., `medlineplus_topics_xml.zip`).
    - Extract the main XML content file (e.g., `pubs.xml` or `mplus_topics_YYYY-MM-DD.xml`).
    - Place this XML file into the `backend/data/medlineplus/` directory.
    - **Important:** Update the `RAW_XML_PATH` variable in `backend/data_preprocessor.py` to match the exact filename of your downloaded XML.
5.  **Preprocess Data and Build Vector Store:**
    These scripts prepare the data for the RAG system.
    - Run the data preprocessor (cleans and chunks XML data):
      ```bash
      python data_preprocessor.py
      ```
      This creates `processed_health_topics.json`.
    - Build the FAISS vector index:
      ```bash
      python build_vector_store.py
      ```
      This creates `health_topics.index`. _(Note: The `chunk_metadata.json` is generated by `build_vector_store.py` but `main.py` now reconstructs metadata from `processed_health_topics.json` for consistency with BM25)._
6.  **Run the FastAPI Backend:**
    ```bash
    python main.py
    ```
    The backend will start, loading all models (this may take a few minutes on first run due to model downloads). It will be available at `http://localhost:8000`. API documentation (Swagger UI) is at `http://localhost:8000/docs`.

**Frontend Setup:**

1.  **Navigate to the frontend directory:**
    ```bash
    # From the project root:
    cd frontend
    # Or from backend: cd ../frontend
    ```
2.  **Install dependencies:**
    ```bash
    yarn install
    # or npm install
    ```
3.  **Set up environment variables:**
    Create a `.env` file in the `frontend` root directory (e.g., `frontend/.env`):
    ```
    VITE_API_URL=http://localhost:8000
    ```
4.  **Run the frontend development server:**
    ```bash
    yarn dev
    # or npm run dev
    ```
    The frontend will typically be available at `http://localhost:5173` (or another port if 5173 is busy).

## ⚠️ Project Scope & Limitations

This project serves as a comprehensive demonstration of building an AI-powered information retrieval system with multimodal capabilities. It has been developed as a learning and portfolio piece.

- **Dataset Constraints:** The accuracy and breadth of answers are inherently limited by the content of the processed MedlinePlus XML data. MedlinePlus is a valuable resource, but it may not cover every specific medical query or the very latest research with the same depth as a continuously updated, specialized medical database. Therefore, for some niche or highly specific queries, the system may not find optimally relevant information.
- **Not a Substitute for Professional Advice:** The information provided by MediQuery AI is for general knowledge and informational purposes only, and does not constitute medical advice, diagnosis, or treatment.
- **Ongoing Development:** This project focuses on the core AI pipeline and backend. Further enhancements in areas like user authentication, conversational context, and advanced error handling would be needed for a production-grade application.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please feel free to check the [issues page](<your-repo-url>/issues) if you have one.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
_(Remember to create a `LICENSE` file in your repository root containing the MIT License text if you choose this license)._

---

**Disclaimer: MediQuery AI is an educational project and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or the health of others.**
