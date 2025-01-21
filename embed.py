#!/usr/bin/env python3

import os
import sys
import torch
import chromadb
from transformers import AutoTokenizer, AutoModel

###############################################################################
#                              CONFIGURATION                                  #
###############################################################################
MODEL_NAME       = "intfloat/e5-large-v2"
INPUT_FILE       = "cleaned.txt"
CHUNK_SIZE       = 512   # max tokens per chunk
OVERLAP_SIZE     = 50    # overlap for successive overflow chunks
BATCH_SIZE       = 8     # Adjust for memory usage vs. speed
PERSIST_DIR      = "./chroma_db"
COLLECTION_NAME  = "text_chunks"

###############################################################################
#                           DEVICE SELECTION                                  #
###############################################################################
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[INFO] Using MPS (Apple Silicon) device for inference.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("[INFO] Using CUDA device for inference.")
else:
    device = torch.device("cpu")
    print("[INFO] Using CPU for inference (may be slower).")

###############################################################################
#                        LOAD MODEL + TOKENIZER                               #
###############################################################################
print("[INFO] Loading tokenizer and model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME)
model.eval().to(device)

###############################################################################
#                    INIT CHROMA (PersistentClient)                           #
###############################################################################
# For Chroma 0.4+, we use PersistentClient. No need to call `.persist()`.
print("[INFO] Initializing persistent ChromaDB client at:", PERSIST_DIR)
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

###############################################################################
#                           HELPER FUNCTIONS                                  #
###############################################################################
def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on the transformer output."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_texts(texts):
    """
    Embeds a list of strings using the loaded e5 model.
    Returns a list of embedding vectors as Python lists.
    """
    # e5 models often use "query:" or "passage:" prefix
    batch_texts = [f"passage: {t}" for t in texts]

    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=CHUNK_SIZE,  # 512
        return_tensors="pt"
    )
    for k in inputs:
        inputs[k] = inputs[k].to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pooled = mean_pooling(outputs, inputs["attention_mask"])
    embeddings = pooled.cpu().numpy().tolist()
    return embeddings

###############################################################################
#                                 MAIN                                        #
###############################################################################
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File '{INPUT_FILE}' not found.")
        sys.exit(1)

    print("[INFO] Processing file in a memory-efficient manner...")
    print(f"[INFO] Input File: {INPUT_FILE}")
    print(f"[INFO] Chunk Size: {CHUNK_SIZE}, Overlap: {OVERLAP_SIZE}")

    # Counters and buffers
    chunk_counter = 0
    batch_texts = []
    batch_ids = []
    batch_metadatas = []

    # Function to flush the current batch to Chroma
    def flush_batch():
        if not batch_texts:
            return
        print(f"[INFO] Embedding & storing {len(batch_texts)} chunks to ChromaDB...")

        emb = embed_texts(batch_texts)

        # Add them to Chroma
        collection.add(
            documents=batch_texts,
            embeddings=emb,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

        print("[INFO] Batch stored (auto-persisted).")

        batch_texts.clear()
        batch_ids.clear()
        batch_metadatas.clear()

    # Read line-by-line
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # **Key**: Tokenize the entire line with overflow turned on
            # So if line is too long, it will produce multiple 512-token chunks
            tokenized = tokenizer(
                line,
                add_special_tokens=False,
                return_overflowing_tokens=True,
                truncation=True,
                max_length=CHUNK_SIZE,
                stride=OVERLAP_SIZE
            )
            # tokenized["input_ids"] is now a list of sub-chunks

            # Convert each sub-chunk of tokens back to text
            chunks = tokenized["input_ids"]
            for chunk_ids in chunks:
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

                batch_texts.append(chunk_text)
                batch_ids.append(f"chunk_{chunk_counter}")
                batch_metadatas.append({
                    "chunk_index": chunk_counter,
                    "source_line_index": line_idx
                })
                chunk_counter += 1

                # If we have enough chunks in memory, flush
                if len(batch_texts) >= BATCH_SIZE:
                    flush_batch()

    # Flush final
    flush_batch()

    print(f"[INFO] Done! Total chunks processed: {chunk_counter}")
    print(f"[INFO] Stored in ChromaDB at: {PERSIST_DIR}")

if __name__ == "__main__":
    main()
