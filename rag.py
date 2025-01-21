#!/usr/bin/env python3

import os
import sys
import torch
import chromadb
import openai
from transformers import AutoTokenizer, AutoModel

###############################################################################
#                              CONFIGURATIONS                                 #
###############################################################################
MODEL_NAME      = "intfloat/e5-large-v2"
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "text_chunks"

OPENAI_API_KEY  = "sqy27-"  # Replace with your real key
GPT_MODEL       = "gpt-4o"

###############################################################################
#                           OPENAI SETUP                                      #
###############################################################################
openai.api_key = OPENAI_API_KEY

###############################################################################
#                           TORCH DEVICE SELECTION                            #
###############################################################################
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[INFO] Using MPS (Apple Silicon).")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("[INFO] Using CUDA.")
else:
    device = torch.device("cpu")
    print("[INFO] Using CPU.")

###############################################################################
#                       LOAD E5 MODEL + TOKENIZER                             #
###############################################################################
print("[INFO] Loading E5 model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval().to(device)

###############################################################################
#                       INIT CHROMA (PersistentClient)                        #
###############################################################################
print(f"[INFO] Initializing ChromaDB at: {CHROMA_DB_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

print(f"[INFO] Getting collection: {COLLECTION_NAME}")
collection = chroma_client.get_collection(name=COLLECTION_NAME)

###############################################################################
#                           HELPER FUNCTIONS                                  #
###############################################################################
def mean_pooling(model_output, attention_mask):
    """Perform mean pooling (E5 style)."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_query(query_text: str):
    """
    Embed a user query with e5, prefix 'query:' as recommended.
    """
    text = f"query: {query_text}"
    inputs = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    for k in inputs:
        inputs[k] = inputs[k].to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    embedding = mean_pooling(outputs, inputs["attention_mask"])
    return embedding.cpu().numpy().tolist()[0]

def build_messages(system_instructions: str, context_chunks, user_query: str):
    """
    Create messages for openai.ChatCompletion (old 0.x syntax).
    'system' gets the instructions + context, 'user' gets the query.
    """
    context_text = ""
    for i, chunk in enumerate(context_chunks, start=1):
        context_text += f"\n--- Chunk {i} ---\n{chunk}\n"

    system_content = (
        f"{system_instructions}\n\n"
        "You have the following context to answer the user's query:\n"
        f"{context_text}\n\n"
        "If the context does not contain the info, say so."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query}
    ]
    return messages

def call_gpt4(messages):
    """
    Call GPT-4 (or GPT-3.5) using openai==0.28.0 syntax with ChatCompletion.
    """
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message["content"]

###############################################################################
#                                   MAIN                                      #
###############################################################################
def main():
    print("\n=============================================")
    print("   Retrieval-Augmented Generation (RAG)      ")
    print("   Using openai==0.28.0 (old ChatCompletion) ")
    print("=============================================")

    # 1. How many chunks to retrieve
    n_str = input("How many chunks to retrieve? [default=5]: ")
    try:
        n = int(n_str)
    except ValueError:
        n = 5

    # 2. User query
    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("[ERROR] Query cannot be empty.")
        return

    # 3. Optional system instructions
    system_instructions = input("Enter system instructions (optional):\n").strip()
    if not system_instructions:
        system_instructions = "You are a helpful assistant with expertise in research."

    # 4. Embed the user query
    print("[INFO] Embedding user query with e5 model...")
    query_embedding = embed_query(user_query)

    # 5. Retrieve from Chroma
    print("[INFO] Querying Chroma for top-n chunks...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    distances = results["distances"][0]
    metas     = results["metadatas"][0]

    print("\nRetrieved Chunks:")
    for i, (doc, dist, meta) in enumerate(zip(docs, distances, metas), start=1):
        print(f"\n--- Chunk {i} ---")
        print("Distance:", dist)
        print("Metadata:", meta)
        print("Content (partial):", doc[:200], "..." if len(doc) > 200 else "")

    # 6. Build messages
    messages = build_messages(system_instructions, docs, user_query)

    # 7. Call GPT-4
    print("\n[INFO] Calling GPT-4 with retrieved context...")
    answer = call_gpt4(messages)

    # 8. Print answer
    print("\n=============================================")
    print("            GPT-4 Response")
    print("=============================================")
    print(answer)
    print()

if __name__ == "__main__":
    main()
