# libguides-rag

# A Home-Grown Approach to Solving Some Basic Problems with WARC-GPT Results

Using Claude (3.5 Sonnet) and ChatGPT (GPT-4o and -o1) for support, I created several Python scripts to optimize the WARC-GPT RAG workflow and improve inference on a collection of WARC files from the same source. The goal was to get cleaner data into the vector store for more effective inference, leveraging existing technologies like:

- `wget` to capture source material from the web,
- `BeautifulSoup` to parse meaningful text,
- A transformer model (`intfloat/e5-large-v2`) for creating embeddings,
- `ChromaDB` for storing embeddings and metadata,
- GPT-4 via OpenAI’s API for inference on embedded chunks and pre-prompts.

---

## Approach Overview

### 1. Get the Data from the Web

The `wget.py` script downloads a website and saves it as a WARC (Web ARChive) file while excluding specific file types for selective archiving. Key features:

- Downloads web content and saves it as a WARC file.
- Excludes predefined file types (e.g., images, videos, fonts) for cleaner archives.
- Utilizes Python's `subprocess` module to execute `wget` commands.

This approach ensures lightweight, focused archives optimized for embedding high-dimensional vectors, contrasting with Archive-It’s approach of preserving entire web pages for human viewing.

---

### 2. Scrape the Good Stuff

The `scrape.py` script extracts meaningful text from WARC files and organizes it into a structured format:

- **Key Functions**:
  - `is_informative`: Filters out noisy or uninformative text.
  - `is_desired_language`: Ensures text is in the desired language (default: English).
  - `extract_text_with_headings`: Groups text under headings for contextual clarity.
  - `chunk_text`: Breaks text into manageable chunks (default: 500 tokens).

The output is saved to `scraped.txt`, containing semantically grouped text organized under headings, ideal for analysis or further processing.

---

### 3. Clean Everything Up

The `clean.py` script removes unwanted characters and normalizes text structure:

- **Functionality**:
  - Removes special characters, excessive whitespace, and unnecessary newlines.
  - Preserves alphanumeric characters, punctuation, and meaningful formatting.

The cleaned output is saved as `cleaned.txt`, providing a more structured and readable text file for subsequent steps.

---

### 4. Count Words and Estimate Tokens

The `count.py` script calculates word counts and estimates token counts:

- **Functions**:
  - `count_words_in_file`: Counts words in the input file.
  - `estimate_tokens`: Estimates token count based on a scaling factor (default: 1.2 tokens per word).

This script is useful for determining if a text file fits within the context window of models like GPT-4 or Gemini.

---

### 5. Create High-Dimensional Vector Embeddings

The `embed.py` script generates vector embeddings from cleaned text:

- **Key Features**:
  - Uses Hugging Face’s `intfloat/e5-large-v2` model for embedding.
  - Splits text into 512-token chunks with 50-token overlap.
  - Stores embeddings and metadata in a persistent `ChromaDB` database.

Embeddings are saved in `./chroma_db`, enabling efficient semantic search.

---

### 6. Retrieve Semantically Similar Text Chunks Based on a User Query

The `query.py` script provides an interface for querying a `ChromaDB` vector database:

- **Process**:
  - Converts user queries into embedding vectors.
  - Searches for the most similar text chunks in the database.
  - Displays top results with metadata and similarity scores.

This script facilitates interactive exploration of stored textual data.

---

### 7. Implement a RAG Pipeline to GPT-4 for Full Inference

The `rag.py` script combines semantic search with GPT-4 for contextually informed responses:

- **Workflow**:
  - Embeds user queries with the `intfloat/e5-large-v2` model.
  - Retrieves relevant text chunks from `ChromaDB`.
  - Sends retrieved context and user query to GPT-4 via OpenAI’s API.
  - Outputs GPT-4’s context-enriched response along with retrieved chunks.

This pipeline leverages `ChromaDB` for efficient vector search and GPT-4’s language capabilities, making it effective for knowledge-based responses, research, or chatbot interactions.
