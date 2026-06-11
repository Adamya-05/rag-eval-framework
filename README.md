RAG Evaluation Framework

A systematic evaluation framework for Retrieval-Augmented Generation (RAG) pipelines, built to compare chunking strategies, embedding models, vector databases, and LLMs  giving a clear picture of which configuration performs best for document Q&A tasks.


What This Does

Most RAG projects pick one configuration and call it done. This framework runs 8 parallel configurations simultaneously and evaluates them against custom metrics, so you can make informed decisions instead of guessing.

What gets compared:

DimensionOptionsChunk sizes300 tokens, 500 tokensEmbedding modelsall-MiniLM-L6-v2, all-mpnet-base-v2Vector databasesFAISS, ChromaDBLLMs (generation)Llama 3.1 8B (via Groq), GPT


Pipeline Architecture

PDF Documents
      ↓
Text Cleaning  (regex-based normalisation)
      ↓
Chunking  (RecursiveCharacterTextSplitter, 2 sizes)
      ↓
Embedding  (HuggingFace sentence-transformers, 2 models)
      ↓
Vector Store  (FAISS + ChromaDB, built in parallel)
      ↓
Retrieval  (top-k similarity search)
      ↓
Generation  (Groq API — Llama 3.1 / GPT)
      ↓
Evaluation  (context relevance + faithfulness scores)


Evaluation Metrics

Context Relevance Score — cosine similarity between the query embedding and each retrieved chunk embedding, averaged across top-k results. Measures how well the retriever is pulling relevant content.

Faithfulness Score — lexical overlap between the generated answer and the retrieved context. Measures whether the LLM is grounding its response in the provided chunks or hallucinating.

Response Time — wall-clock latency per model, benchmarked across identical inputs for fair comparison.


Tech Stack


LangChain — document loading, chunking, retrieval orchestration
HuggingFace sentence-transformers — local embedding models (no API cost)
FAISS — in-memory vector search (Facebook AI)
ChromaDB — persistent vector database
Groq API — fast LLM inference (Llama 3.1 8B Instant)
Google Colab — GPU-accelerated execution environment
PyPDF / LangChain PDF Loader — document ingestion



Setup

1. Clone the repo

bashgit clone https://github.com/your-username/rag-eval-framework.git
cd rag-eval-framework

2. Install dependencies

bashpip install langchain langchain-community langchain-text-splitters
pip install faiss-cpu chromadb sentence-transformers
pip install pypdf groq python-dotenv

3. Set your API key

Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key_here

Get a free Groq API key at console.groq.com.


Never hardcode API keys in notebooks. Load them via os.environ.get("GROQ_API_KEY").



4. Add your documents

Upload PDF files to the /content directory (or update the folder_path variable in the notebook to point to your local PDF folder).

5. Run the notebook

Open Rag_assignment_GenAI.ipynb in Jupyter or Google Colab and run all cells.


Configuration

All key parameters are defined in a single config block at the top of the notebook:

pythonCHUNK_SIZES = [300, 500]
CHUNK_OVERLAPS = [50]
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]
VECTOR_DBS = ["faiss", "chroma"]
TOP_K = 3

Swap in any HuggingFace sentence-transformer model or any Groq-supported LLM by editing these values.


Sample Output

The framework outputs a structured comparison across all configurations:

--- faiss_all-MiniLM-L6-v2_500 ---
Result 1: Normalization is the process of organising data in a database...
Result 2: First Normal Form (1NF) requires that each column contains atomic values...

MODEL: llama-3.1-8b-instant
Response Time: 1.83 sec
Answer Length: 142 words

📊 Context Relevance: 0.847
📊 Faithfulness: 0.631


Project Structure

rag-eval-framework/
├── Rag_assignment_GenAI.ipynb   # Main notebook
├── .env                          # API keys (not committed)
├── .gitignore
└── README.md


Key Learnings


Smaller chunk sizes (300) improve context relevance scores for factual Q&A but increase retrieval noise for broader questions
all-mpnet-base-v2 produces higher relevance scores than all-MiniLM-L6-v2 at the cost of ~2x slower embedding generation
FAISS and ChromaDB return near-identical top-k results for semantic similarity tasks at this scale; ChromaDB adds persistence which matters for production use
Llama 3.1 8B via Groq is significantly faster than GPT alternatives with comparable faithfulness scores for domain-specific Q&A



Author

Adamya Agarwal
B.Tech CSE, BML Munjal University (2023–2027)
