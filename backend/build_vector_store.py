# backend/build_vector_store.py
import json
import os
import faiss  # For vector similarity search
from sentence_transformers import SentenceTransformer
import numpy as np
import torch  # For checking CUDA availability

# --- Configuration ---
PROCESSED_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "data", "medlineplus", "processed_health_topics.json"
)
FAISS_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "data", "medlineplus", "health_topics.index"
)
CHUNK_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "medlineplus", "chunk_metadata.json"
)  # To store ID -> chunk mapping

# Select Sentence Transformer model
# Ensure this matches the model loaded in main.py for consistency if you generate query embeddings there
MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-tas-b"


def build_faiss_index():
    print("Starting FAISS index build process...")

    # 1. Load processed data
    try:
        with open(PROCESSED_JSON_PATH, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Processed data file not found at {PROCESSED_JSON_PATH}.")
        print("Please run data_preprocessor.py first.")
        return

    if not processed_data:
        print("No data found in processed JSON. Exiting.")
        return

    print(f"Loaded {len(processed_data)} text chunks for processing.")

    # 2. Initialize Sentence Transformer model
    print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        print(f"Failed to load Sentence Transformer model: {e}")
        return
    print("Sentence Transformer model loaded.")

    # 3. Generate embeddings for all chunks
    print("Generating embeddings for all text chunks (this may take a while)...")
    # Extract the 'content_chunk' text from each dictionary
    chunk_texts = [item["content_chunk"] for item in processed_data]

    # It's good to show progress for long operations
    batch_size = 32  # Adjust batch size based on your RAM/VRAM
    all_embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        embeddings = model.encode(
            batch, convert_to_tensor=False, show_progress_bar=False
        )  # convert_to_tensor=False for numpy
        all_embeddings.extend(embeddings)
        print(
            f"Processed batch {i//batch_size + 1}/{(len(chunk_texts) -1)//batch_size + 1}"
        )

    if not all_embeddings:
        print("No embeddings were generated. Check your data and model.")
        return

    embeddings_np = np.array(all_embeddings).astype(
        "float32"
    )  # FAISS requires float32 numpy arrays
    print(
        f"Generated {embeddings_np.shape[0]} embeddings with dimension {embeddings_np.shape[1]}."
    )

    # 4. Build FAISS index
    dimension = embeddings_np.shape[1]  # Dimension of embeddings
    # Using IndexFlatL2 - a simple L2 distance index. For larger datasets, consider more advanced indexes.
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)  # Add embeddings to the index
    print(f"FAISS index built. Total vectors in index: {index.ntotal}")

    # 5. Save the FAISS index and chunk metadata
    # Ensure the directory for the output files exists
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")

    # Save metadata (mapping index position to original chunk ID or full data)
    # We need this to retrieve the actual text chunk after finding its embedding's index
    chunk_metadata = {i: item for i, item in enumerate(processed_data)}
    with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=4)
    print(f"Chunk metadata saved to: {CHUNK_METADATA_PATH}")

    print("FAISS index build process completed successfully.")


if __name__ == "__main__":
    build_faiss_index()
