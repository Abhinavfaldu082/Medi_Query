# backend/main.py
from fastapi.responses import StreamingResponse  # To stream the PDF
import io  # For in-memory PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from contextlib import asynccontextmanager
import json
import html
import re  # For improved tokenization

# --- AI Model Imports ---
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import whisper
import easyocr
import torch
import faiss
import numpy as np
from rank_bm25 import BM25Okapi  # For Keyword Search

# --- Global Variables for Models & RAG Components ---
qa_pipeline = None
summarization_pipeline = None
whisper_model = None
easyocr_reader = None
sentence_transformer_model = None
faiss_index = None
chunk_metadata = None
bm25_index = None  # For BM25 keyword search
tokenized_corpus_for_bm25 = None  # Store tokenized text for BM25
full_processed_data_for_bm25 = None  # Store the original chunk data for BM25 retrieval
zero_shot_classifier = None
SYMPTOM_CANDIDATE_LABELS = [  # Define a list of common symptom categories
    "headache",
    "migraine",
    "fever",
    "chills",
    "cough",
    "sore throat",
    "runny nose",
    "congestion",
    "shortness of breath",
    "difficulty breathing",
    "chest pain",
    "fatigue",
    "weakness",
    "dizziness",
    "nausea",
    "vomiting",
    "diarrhea",
    "constipation",
    "abdominal pain",
    "stomach ache",
    "skin rash",
    "joint pain",
    "muscle pain",
    "back pain",
    "anxiety",
    "depression",
    "insomnia",
    "memory loss",
    "confusion",
    "blurry vision",
    # Add more as needed, be reasonably broad but not overly granular initially
]

# --- Configuration ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
]

UPLOAD_DIRECTORY = "temp_uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "medlineplus")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "health_topics.index")
CHUNK_METADATA_PATH = os.path.join(DATA_DIR, "chunk_metadata.json")
RAG_TOP_K = 5


# --- Tokenizer for BM25 ---
def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    # Optionally, add stop word removal here if needed for your corpus
    # from nltk.corpus import stopwords
    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in text.split() if word not in stop_words]
    # return tokens
    return text.split()


# --- Model Loading Function (Updated) ---
def load_models_and_rag_components():
    global qa_pipeline, summarization_pipeline, whisper_model, easyocr_reader
    global sentence_transformer_model, faiss_index, chunk_metadata
    global bm25_index, tokenized_corpus_for_bm25, full_processed_data_for_bm25
    global zero_shot_classifier
    
    print("Loading AI models and RAG components...")
    device_hf_pipeline = 0 if torch.cuda.is_available() else -1
    device_sbert = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Question Answering Model
    print("Loading Question Answering model...")
    qa_model_name = "dmis-lab/biobert-base-cased-v1.1-squad"
    try:
        qa_pipeline = pipeline(
            "question-answering",
            model=qa_model_name,
            tokenizer=qa_model_name,
            device=device_hf_pipeline,
        )
        print(f"QA model '{qa_model_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading QA model '{qa_model_name}': {e}")

    # 2. Summarization Model
    print("Loading Summarization model...")
    summarization_model_name = "sshleifer/distilbart-cnn-12-6"
    try:
        summarization_pipeline = pipeline(
            "summarization",
            model=summarization_model_name,
            tokenizer=summarization_model_name,
            device=device_hf_pipeline,
        )
        print(f"Summarization model '{summarization_model_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading Summarization model '{summarization_model_name}': {e}")

    # 3. Whisper Model
    print("Loading Whisper model (Speech-to-Text)...")
    whisper_model_name = "base"
    try:
        whisper_model = whisper.load_model(whisper_model_name, device=device_sbert)
        print(f"Whisper model '{whisper_model_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading Whisper model '{whisper_model_name}': {e}")

    # 4. EasyOCR Reader
    print("Loading EasyOCR model (OCR)...")
    try:
        easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        print("EasyOCR model loaded successfully.")
    except Exception as e:
        print(f"Error loading EasyOCR model: {e}")

    # 5. Sentence Transformer Model
    print("Loading Sentence Transformer model (for RAG)...")
    sentence_transformer_model_name = "sentence-transformers/msmarco-distilbert-base-tas-b"  # Kept this from previous step
    try:
        sentence_transformer_model = SentenceTransformer(
            sentence_transformer_model_name, device=device_sbert
        )
        print(
            f"Sentence Transformer model '{sentence_transformer_model_name}' loaded successfully."
        )
    except Exception as e:
        print(
            f"Error loading Sentence Transformer model '{sentence_transformer_model_name}': {e}"
        )

    # 6. Load FAISS Index
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            print(
                f"FAISS index loaded. Total vectors: {faiss_index.ntotal if faiss_index else 'None'}"
            )
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            faiss_index = None
    else:
        print(
            f"FAISS index file not found at {FAISS_INDEX_PATH}. RAG Q&A will be limited (FAISS part)."
        )
        faiss_index = None

    # 7. Load Chunk Metadata (used by both FAISS retrieval and BM25 prep)
    # And also load full_processed_data for BM25 corpus text.
    # Ideally, chunk_metadata would just be indices and we'd load the full processed_health_topics.json for text
    # For now, assuming chunk_metadata values are the full chunk dicts OR we load the main JSON.
    # Let's load the main JSON to be sure for BM25 data source.

    processed_data_path = os.path.join(
        DATA_DIR, "processed_health_topics.json"
    )  # Path to the JSON from data_preprocessor.py
    if os.path.exists(processed_data_path):
        print(
            f"Loading full processed data from: {processed_data_path} for BM25 and metadata linking."
        )
        try:
            with open(processed_data_path, "r", encoding="utf-8") as f:
                full_processed_data_for_bm25 = json.load(f)  # This is a list of dicts

            # Prepare chunk_metadata (mapping FAISS index to original chunk data)
            # This assumes FAISS was built on `full_processed_data_for_bm25` in that order.
            chunk_metadata = {
                i: item for i, item in enumerate(full_processed_data_for_bm25)
            }
            print(
                f"Chunk metadata prepared from full processed data. Total items: {len(chunk_metadata)}"
            )

            # 8. Prepare for BM25 Keyword Search
            print(f"Preparing BM25 index from full processed data...")
            corpus_texts = [
                item["content_chunk"] for item in full_processed_data_for_bm25
            ]
            tokenized_corpus_for_bm25 = [simple_tokenizer(doc) for doc in corpus_texts]

            # --- DEBUG: Inspecting CPR documents for BM25 (KEEP THIS DEBUG BLOCK) ---
            print(
                "--- DEBUG: Inspecting CPR documents for BM25 (after improved tokenization) ---"
            )
            cpr_doc_indices_in_bm25_corpus = []
            for i, doc_data_item in enumerate(
                full_processed_data_for_bm25
            ):  # Iterate over the list
                if doc_data_item.get("title", "").lower() == "cpr":
                    cpr_doc_indices_in_bm25_corpus.append(i)
                    print(
                        f"Found CPR doc for BM25 at index {i}, Title: {doc_data_item.get('title')}"
                    )
                    if i < len(tokenized_corpus_for_bm25):
                        print(
                            f"Tokenized CPR Content (first 20 tokens): {tokenized_corpus_for_bm25[i][:20]}"
                        )
                    else:
                        print(
                            f"ERROR: Index {i} out of bounds for tokenized_corpus_for_bm25 (len: {len(tokenized_corpus_for_bm25)})"
                        )
            if not cpr_doc_indices_in_bm25_corpus:
                print(
                    "WARNING: No documents with title 'CPR' found in full_processed_data_for_bm25!"
                )
            print("--- END BM25 CPR Document Inspection ---")

            bm25_index = BM25Okapi(tokenized_corpus_for_bm25)
            print(
                f"BM25 index prepared with {len(tokenized_corpus_for_bm25)} documents."
            )

        except Exception as e:
            print(f"Error loading full processed data or preparing BM25 index: {e}")
            full_processed_data_for_bm25 = None
            chunk_metadata = (
                None  # If full data load fails, metadata from it also fails
            )
            bm25_index = None
    else:
        print(
            f"Full processed data file not found at {processed_data_path}. BM25 and FAISS metadata linking will fail."
        )
        full_processed_data_for_bm25 = None
        chunk_metadata = None
        bm25_index = None

    # Within load_models_and_rag_components()
    # ... after other models ...
    # 9. Zero-Shot Classifier (for symptom keyword identification)
    print("Loading Zero-Shot Classification model...")
    # Using a common model suitable for NLI tasks which zero-shot often relies on
    zero_shot_model_name = "facebook/bart-large-mnli"
    try:
        zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=zero_shot_model_name,
            device=device_hf_pipeline,
        )
        print(
            f"Zero-Shot Classification model '{zero_shot_model_name}' loaded successfully."
        )
    except Exception as e:
        print(f"Error loading Zero-Shot Classification model '{zero_shot_model_name}': {e}")
    print("AI models and RAG components loading complete.")


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Initializing models and RAG components...")
    load_models_and_rag_components()
    print("Application startup: Initialization complete.")
    yield
    print("Application shutdown: Cleaning up...")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="MediQuery AI API",
    description="API for multimodal symptom triage and health search assistant.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Root Endpoint ---
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to MediQuery AI API! Models and RAG components should be loaded."
    }


# --- Helper function for RAG ---
def retrieve_relevant_chunks_hybrid(
    question,
    question_embedding,
    top_k=RAG_TOP_K,
    k_rrf=60,  # k_rrf is a constant for RRF, often 60
):
    candidate_docs = {}
    # 1. FAISS Vector Search - Get top N candidates (e.g., top_k * 5 or a fixed larger number)
    num_faiss_candidates = top_k * 10
    if faiss_index and chunk_metadata:
        query_vector = np.array([question_embedding], dtype="float32")
        try:
            distances, faiss_indices = faiss_index.search(
                query_vector, num_faiss_candidates
            )
            for rank, idx in enumerate(faiss_indices[0]):
                if idx != -1 and idx in chunk_metadata:
                    doc_data = chunk_metadata[idx]
                    doc_id = doc_data["id"]
                    if doc_id not in candidate_docs:
                        candidate_docs[doc_id] = {
                            "data": doc_data,
                            "faiss_rank": None,
                            "bm25_rank": None,
                            "rrf_score": 0.0,
                        }
                    candidate_docs[doc_id]["faiss_rank"] = rank + 1  # Rank is 1-based
        except Exception as e:
            print(f"Error during FAISS search: {e}")

    # 2. BM25 Keyword Search - Get top N candidates
    num_bm25_candidates = top_k * 10
    if bm25_index and full_processed_data_for_bm25:
        tokenized_query = simple_tokenizer(question)
        try:
            bm25_scores = bm25_index.get_scores(tokenized_query)
            scored_indices = sorted(
                [
                    (score, original_idx)
                    for original_idx, score in enumerate(bm25_scores)
                ],
                key=lambda x: x[0],
                reverse=True,
            )
            for rank, (score, original_bm25_idx) in enumerate(
                scored_indices[:num_bm25_candidates]
            ):
                if score > 0.0:  # Only consider actual matches
                    doc_data = full_processed_data_for_bm25[original_bm25_idx]
                    doc_id = doc_data["id"]
                    if doc_id not in candidate_docs:
                        candidate_docs[doc_id] = {
                            "data": doc_data,
                            "faiss_rank": None,
                            "bm25_rank": None,
                            "rrf_score": 0.0,
                        }
                    candidate_docs[doc_id]["bm25_rank"] = rank + 1  # Rank is 1-based
        except Exception as e:
            print(f"Error during BM25 search: {e}")

    # 3. Calculate RRF Score for each document that has at least one rank
    for doc_id, ranks_data in candidate_docs.items():
        rrf_score = 0.0
        if ranks_data["faiss_rank"] is not None:
            rrf_score += 1.0 / (k_rrf + ranks_data["faiss_rank"])
        if ranks_data["bm25_rank"] is not None:
            rrf_score += 1.0 / (k_rrf + ranks_data["bm25_rank"])
        candidate_docs[doc_id]["rrf_score"] = rrf_score

    # 4. Sort by RRF score
    sorted_docs_by_rrf = sorted(
        candidate_docs.values(), key=lambda x: x["rrf_score"], reverse=True
    )

    # 5. Return the data of the actual top_k documents
    final_retrieved_data = [doc["data"] for doc in sorted_docs_by_rrf[:top_k]]

    print(
        f"Hybrid retrieval (RRF): Found {len(final_retrieved_data)} chunks for context."
    )
    if final_retrieved_data:
        print(
            f"Titles from RRF hybrid: {[d.get('title', 'N/A') for d in final_retrieved_data]}"
        )

    return final_retrieved_data

# --- API Endpoints ---
# ... (process_text_input, process_image_input, process_audio_input, summarize_text are the same as before) ...


@app.post("/text_input")
async def process_text_input(text: str = Form(...)):
    if not qa_pipeline:
        raise HTTPException(
            status_code=503, detail="Question Answering model is not available."
        )
    print(f"Received text for general processing: {text}")
    return {
        "received_text": text,
        "message": "Text input received. For full Q&A, use the /qa endpoint.",
    }


@app.post("/image_input")
async def process_image_input(image_file: UploadFile = File(...)):
    if not easyocr_reader:
        raise HTTPException(status_code=503, detail="OCR model is not available.")
    file_path = os.path.join(UPLOAD_DIRECTORY, image_file.filename)
    text_results = []
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        ocr_result = easyocr_reader.readtext(file_path)
        for bbox, text, prob in ocr_result:
            text_results.append({"text": text, "confidence": float(prob)})
        return {"filename": image_file.filename, "ocr_results": text_results}
    except Exception as e:
        print(f"OCR Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing image with OCR: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/audio_input")
async def process_audio_input(audio_file: UploadFile = File(...)):
    if not whisper_model:
        raise HTTPException(
            status_code=503, detail="Speech-to-Text model is not available."
        )
    file_path = os.path.join(UPLOAD_DIRECTORY, audio_file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        transcription_result = whisper_model.transcribe(
            file_path, fp16=(torch.cuda.is_available())
        )
        return {
            "filename": audio_file.filename,
            "transcription": transcription_result["text"],
            "language": transcription_result["language"],
        }
    except Exception as e:
        print(f"Transcription Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error transcribing audio: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/summarize_text")
async def summarize_text_input(text_to_summarize: str = Form(...)):
    if not summarization_pipeline:
        raise HTTPException(
            status_code=503, detail="Summarization model is not available."
        )
    try:
        summary = summarization_pipeline(
            text_to_summarize, max_length=150, min_length=30, do_sample=False
        )
        return {
            "original_text_preview": text_to_summarize[:200] + "...",
            "summary": summary[0]["summary_text"],
        }
    except Exception as e:
        print(f"Summarization Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error during summarization: {str(e)}"
        )


@app.post("/qa")
async def health_question_answering(question: str = Form(...)):
    if (
        not sentence_transformer_model
    ):  # Needed for question embedding even if BM25 is primary
        raise HTTPException(
            status_code=503, detail="Sentence embedding model not available."
        )
    if not qa_pipeline:
        raise HTTPException(
            status_code=503, detail="Question Answering model not available."
        )
    # No direct check for bm25_index/faiss_index here as retrieve_relevant_chunks_hybrid handles it

    try:
        print(f"Hybrid RAG Q&A for question: {question}")  # Stays as is
        question_embedding = sentence_transformer_model.encode(
            question, convert_to_tensor=False
        )

        relevant_chunks_data = retrieve_relevant_chunks_hybrid(
            question, question_embedding, top_k=RAG_TOP_K
        )

        if not relevant_chunks_data:
            print("No relevant chunks found for the question (BM25 ALONE test).")
            return {
                "question": question,
                "answer": "I couldn't find specific information related to your question in the available knowledge base (BM25 ALONE test). Please try rephrasing.",
                "retrieved_context_preview": [],
                "sources": [],
            }

        context_parts = [
            chunk_data["content_chunk"] for chunk_data in relevant_chunks_data
        ]
        context_for_qa = " ".join(context_parts)

        # ---- TEMPORARY PRINT FOR DEBUGGING CPR ----
        if "cpr" in question.lower():  # Keep this for now
            print("-" * 50)
            print(f"FULL HYBRID CONTEXT FOR '{question}':\n{context_for_qa}")
            print("-" * 50)
        else:
            print(f"Context for QA (first 500 chars): {context_for_qa[:500]}...")

        qa_result = qa_pipeline(question=question, context=context_for_qa)
        answer = qa_result.get(
            "answer", "No answer could be extracted from the context."
        )

        sources = list(
            set(
                [
                    (chunk_data["title"], chunk_data["url"])
                    for chunk_data in relevant_chunks_data
                    if chunk_data.get("url")
                ]
            )
        )
        sources_dict_list = [{"title": title, "url": url} for title, url in sources]

        print(f"Answer found: {answer[:100]}...")
        return {
            "type": "qa",
            "question": question,
            "answer": answer,
            "retrieved_context_preview": [
                chunk["content_chunk"][:150] + "..." for chunk in relevant_chunks_data
            ],
            "sources": sources_dict_list,
        }
    except Exception as e:
        print(f"Error during Q&A processing: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while answering the question: {str(e)}",
        )


@app.post("/symptom_explore")
async def explore_symptoms_endpoint(
    symptoms_text: str = Form(...),
):  # Changed name from explore_symptoms
    if not zero_shot_classifier:
        raise HTTPException(
            status_code=503,
            detail="Symptom analysis model (Zero-Shot) is not available.",
        )
    if (
        not sentence_transformer_model
        or not qa_pipeline
        or not faiss_index
        or not chunk_metadata
    ):
        # Check for RAG components because we will feed results to the QA system
        raise HTTPException(
            status_code=503,
            detail="RAG components not available for symptom exploration.",
        )

    print(f"Exploring symptoms from text: {symptoms_text}")

    try:
        # 1. Identify potential symptom categories using Zero-Shot Classification
        # multi_label=True allows it to pick multiple relevant labels
        classification_results = zero_shot_classifier(
            symptoms_text, SYMPTOM_CANDIDATE_LABELS, multi_label=True
        )

        # Filter for labels with a score above a certain threshold (e.g., 0.7)
        # This threshold needs tuning based on model performance and label set.
        initial_candidates = []
        for label, score in zip(classification_results['labels'], classification_results['scores']):
            if score > 0.9: # Your current high threshold
                initial_candidates.append({"label": label, "score": score})
            else:
                break # Scores are sorted, so no need to check further

        # Take top N (e.g., 3) from these high-confidence candidates
        max_symptoms_to_query = 3 
        identified_symptoms = [candidate['label'] for candidate in initial_candidates[:max_symptoms_to_query]]

        if not identified_symptoms:
            return {
                "input_text": symptoms_text,
                "identified_symptoms": [],
                "exploration_results": [
                    "Could not identify specific symptom categories from your description. Please try rephrasing or being more specific."
                ],
            }

        print(f"Identified symptom keywords/categories: {identified_symptoms}")

        # 2. For each identified symptom, formulate a question and query the RAG Q&A system
        exploration_data = []
        for symptom in identified_symptoms:
            # Formulate a question for the RAG system
            # We could also just use the symptom as the query if the RAG handles keywords well,
            # but a question format is often better.
            query_question = f"What are common causes and information about {symptom}?"

            print(f"Querying RAG for: {query_question}")
            # --- Reusing the RAG Q&A logic ---
            question_embedding = sentence_transformer_model.encode(
                query_question, convert_to_tensor=False
            )
            relevant_chunks_data = retrieve_relevant_chunks_hybrid(
                query_question, question_embedding, top_k=RAG_TOP_K
            )  # Using RAG_TOP_K from global config

            if relevant_chunks_data:
                context_parts = [
                    chunk_data["content_chunk"] for chunk_data in relevant_chunks_data
                ]
                context_for_qa = " ".join(context_parts)

                qa_result = qa_pipeline(question=query_question, context=context_for_qa)
                answer = qa_result.get(
                    "answer",
                    f"Could not find a specific answer for {symptom} in the context.",
                )

                sources = list(
                    set(
                        [
                            (chunk_data["title"], chunk_data["url"])
                            for chunk_data in relevant_chunks_data
                            if chunk_data.get("url")
                        ]
                    )
                )
                sources_dict_list = [
                    {"title": title, "url": url} for title, url in sources
                ]

                exploration_data.append(
                    {
                        "symptom_category": symptom,
                        "related_question_asked": query_question,
                        "information": answer,
                        "sources": sources_dict_list,
                    }
                )
            else:
                exploration_data.append(
                    {
                        "symptom_category": symptom,
                        "related_question_asked": query_question,
                        "information": f"No specific information found for {symptom} using the current knowledge base.",
                        "sources": [],
                    }
                )

        return {
            "type": "symptom_explore",
            "input_text": symptoms_text,
            "identified_symptoms_for_query": identified_symptoms,  # What we used to query RAG
            "exploration_results": exploration_data,
            "disclaimer": "This information is for general knowledge only and NOT a substitute for professional medical advice. Always consult a doctor or qualified healthcare provider for any health concerns or before making any decisions related to your health.",
        }

    except Exception as e:
        print(f"Error during symptom exploration: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during symptom exploration: {str(e)}",
        )


# --- Report Generation Endpoint ---
@app.post("/generate_report")
async def generate_report_endpoint(session_data: dict):
    """
    Generates a PDF report from session data.
    Expects session_data to be a dictionary, potentially matching the output
    structure of /qa or /symptom_explore, or a custom structure.

    Example expected session_data for /qa:
    {
        "type": "qa", // To distinguish report type
        "question": "User's question",
        "answer": "AI's answer",
        "sources": [{"title": "Source Title", "url": "Source URL"}, ...]
        "userInput": "User's original full text input if different from question" // Optional
    }

    Example expected session_data for /symptom_explore:
    {
        "type": "symptom_explore",
        "input_text": "User's symptom description",
        "identified_symptoms_for_query": ["symptom1", "symptom2"],
        "exploration_results": [
            {
                "symptom_category": "symptom1",
                "related_question_asked": "RAG question for symptom1",
                "information": "Answer for symptom1",
                "sources": [{"title": "S1T1", "url": "S1U1"}, ...]
            },
            // ... more symptoms
        ],
        "disclaimer": "Medical disclaimer text"
    }
    """
    print(
        f"Generating report for session_data type: {session_data.get('type', 'Unknown')}"
    )

    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        styles = getSampleStyleSheet()
        story = []

        # 1. Header
        story.append(Paragraph("MediQuery AI Report", styles["h1"]))
        story.append(
            Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.2 * inch))
        story.append(HRFlowable(width="100%", thickness=1, color="grey"))
        story.append(Spacer(1, 0.2 * inch))

        report_type = session_data.get("type", "unknown")

        # 2. User Input Section
        if report_type == "qa":
            user_input_text = session_data.get(
                "userInput", session_data.get("question", "N/A")
            )
            story.append(Paragraph("Your Question:", styles["h2"]))
            story.append(Paragraph(user_input_text, styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))
        elif report_type == "symptom_explore":
            user_input_text = session_data.get("input_text", "N/A")
            story.append(Paragraph("Your Symptom Description:", styles["h2"]))
            story.append(Paragraph(user_input_text, styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
            identified_symptoms = session_data.get("identified_symptoms_for_query", [])
            if identified_symptoms:
                story.append(
                    Paragraph(
                        f"Identified Query Terms: {', '.join(identified_symptoms)}",
                        styles["Italic"],
                    )
                )
            story.append(Spacer(1, 0.2 * inch))
        else:
            story.append(Paragraph("User Input:", styles["h2"]))
            story.append(
                Paragraph(str(session_data.get("input_data", "N/A")), styles["Normal"])
            )  # Generic fallback
            story.append(Spacer(1, 0.2 * inch))

        # 3. AI Response / Information Section
        story.append(Paragraph("Information Found:", styles["h2"]))

        if report_type == "qa":
            story.append(
                Paragraph(
                    f"Answer: {session_data.get('answer', 'N/A')}", styles["Normal"]
                )
            )
            story.append(Spacer(1, 0.1 * inch))
            sources = session_data.get("sources", [])
            if sources:
                story.append(Paragraph("Sources:", styles["h3"]))
                for source in sources:
                    story.append(
                        Paragraph(
                            f"- {source.get('title', 'N/A')} ({source.get('url', 'N/A')})",
                            styles["Bullet"],
                        )
                    )
                story.append(Spacer(1, 0.2 * inch))

        elif report_type == "symptom_explore":
            exploration_results = session_data.get("exploration_results", [])
            for i, result in enumerate(exploration_results):
                story.append(
                    Paragraph(
                        f"Regarding: {result.get('symptom_category', 'N/A')}",
                        styles["h3"],
                    )
                )
                # story.append(Paragraph(f"<i>Queried as: {result.get('related_question_asked', 'N/A')}</i>", styles['Italic'])) # Optional
                story.append(
                    Paragraph(result.get("information", "N/A"), styles["Normal"])
                )
                story.append(Spacer(1, 0.1 * inch))
                sources = result.get("sources", [])
                if sources:
                    story.append(
                        Paragraph("Potential Sources:", styles["BodyText"])
                    )  # Smaller heading for sub-sources
                    for source in sources:
                        story.append(
                            Paragraph(
                                f"  - {source.get('title', 'N/A')} ({source.get('url', 'N/A')})",
                                styles["Bullet"],
                            )
                        )
                story.append(
                    Spacer(1, 0.2 if i < len(exploration_results) - 1 else 0.1 * inch)
                )  # Less space after last item
        else:  # Generic fallback
            story.append(
                Paragraph(
                    str(session_data.get("output_data", "No information provided.")),
                    styles["Normal"],
                )
            )

        # 4. Disclaimer
        story.append(Spacer(1, 0.3 * inch))
        story.append(HRFlowable(width="100%", thickness=0.5, color="lightgrey"))
        story.append(Spacer(1, 0.1 * inch))
        disclaimer_text = session_data.get(
            "disclaimer",
            "This information is for general knowledge only and NOT a substitute for professional medical advice. "
            "Always consult a doctor or qualified healthcare provider for any health concerns or before making "
            "any decisions related to your health.",
        )
        disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",  # Name of the new style
            parent=styles["Normal"],  # Inherit from 'Normal'
            fontSize=8,
            italic=True,
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style)) # Pass the style object directly
        doc.build(story)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=MediQuery_AI_Report.pdf"
            },
        )

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Could not generate PDF report: {str(e)}"
        )


# --- Main Block ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
