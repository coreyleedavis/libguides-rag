#!/usr/bin/env python3

import torch
import chromadb
from transformers import AutoTokenizer, AutoModel

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL_NAME = "intfloat/e5-large-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "text_chunks"

# ----------------------------------------------------------------------------
# Set up device
# ----------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[INFO] Using MPS (Apple Silicon) device for inference.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("[INFO] Using CUDA device for inference.")
else:
    device = torch.device("cpu")
    print("[INFO] Using CPU for inference.")

# ----------------------------------------------------------------------------
# Load model & tokenizer
# ----------------------------------------------------------------------------
print("[INFO] Loading model & tokenizer:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
model.to(device)

# ----------------------------------------------------------------------------
# Initialize Chroma PersistentClient
# ----------------------------------------------------------------------------
# Requires Chroma >= 0.4
print("[INFO] Initializing persistent ChromaDB client at:", CHROMA_DB_PATH)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

print("[INFO] Getting collection:", COLLECTION_NAME)
collection = chroma_client.get_collection(COLLECTION_NAME)

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on the model_output (E5 or similar)."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_query(text: str):
    """
    Embed a user query using the E5 model.
    For queries, e5 recommends prefixing with 'query: '.
    """
    input_text = f"query: {text}"
    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    for k in inputs:
        inputs[k] = inputs[k].to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embedding = mean_pooling(outputs, inputs["attention_mask"])
    return embedding.cpu().numpy().tolist()[0]

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("\n========================")
    print(" Chroma Query Interface")
    print("========================")
    
    # Ask user how many chunks to retrieve
    n_str = input("How many chunks would you like to retrieve? (e.g., 5): ")
    try:
        n = int(n_str)
    except ValueError:
        print("[ERROR] Invalid number. Defaulting to 5.")
        n = 5

    # Ask user for their query
    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("[ERROR] Query cannot be empty. Exiting.")
        return

    # 1) Embed the user's query
    query_vector = embed_query(user_query)

    # 2) Search the collection
    print("[INFO] Searching the collection for the most similar chunks...")
    results = collection.query(
        query_embeddings=[query_vector],  # must be a list of embeddings
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    # 3) Display results
    # results is a dict: { "documents": [...], "metadatas": [...], "distances": [...] }
    # each is a list of lists because we can query multiple embeddings at once.
    documents = results["documents"][0]   # list of strings
    metadatas = results["metadatas"][0]   # list of dicts
    distances = results["distances"][0]   # list of floats

    print("\n========================")
    print(f"   Top {n} Results")
    print("========================")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        print(f"\nResult {i}:")
        print("  Distance: ", dist)
        print("  Chunk ID:", meta.get("chunk_index", "N/A"))
        print("  Metadata:", meta)
        print("  Text:    ", doc)  # Display the entire chunk text

if __name__ == "__main__":
    main()
