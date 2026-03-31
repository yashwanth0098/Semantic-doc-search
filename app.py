import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# -------------------------------
# 1. File paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pdf_files = [
    os.path.join(BASE_DIR, "data", "file2.pdf"),
    os.path.join(BASE_DIR, "data", "file1.pdf")
]

# -------------------------------
# 2. Load PDFs
# -------------------------------
text = ""
for file in pdf_files:
    if os.path.exists(file):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    else:
        print(f"File not found: {file}")

# -------------------------------
# 3. Split text
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)
chunks = splitter.split_text(text)

# -------------------------------
# 4. Embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# 5. Vector store
# -------------------------------
vector_db = FAISS.from_texts(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# -------------------------------
# 6. LLM (Ollama)
# -------------------------------
llm = OllamaLLM(model="llama3")

# -------------------------------
# 7. Ask Question
# -------------------------------
query = input("Ask a question: ")

# Retrieve documents
docs = retriever.invoke(query)

context = "\n\n".join(doc.page_content for doc in docs)

prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

# ✅ Correct invocation
answer = llm.invoke(prompt)

print("\nAnswer:\n", answer)








# src/ingestion.py

from pathlib import Path
import os

from config.path import (
    RAW_PDF_DIR,
    RAW_TEXT_DIR,
    RAW_TABLES_DIR,
    METADATA_DIR,
    create_directories
)

from src.utils.helper import (
    get_logger,
    get_document_client,
    get_pdf_files,
    generate_metadata,
    analyze_document,
    save_metadata,
    save_text,
    save_tables
)

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self):
        try:
            logger.info("Initializing DataIngestion pipeline")

            create_directories()

            self.endpoint = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
            self.key = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

            if not self.endpoint or not self.key:
                raise ValueError("Azure credentials not found in environment variables")

            self.client = get_document_client(self.endpoint, self.key)

            logger.info("Azure Document Intelligence client initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise

    def process_single_pdf(self, pdf_path: Path):
        try:
            logger.info(f"Processing started: {pdf_path.name}")

            # 1. Generate metadata
            metadata = generate_metadata(pdf_path)
            logger.info("Metadata generated")

            # 2. Azure Document Intelligence analysis
            result = analyze_document(self.client, pdf_path)
            logger.info("Document analysis completed")

            file_stem = pdf_path.stem

            # 3. Save outputs
            save_metadata(
                metadata,
                result,
                METADATA_DIR / f"{file_stem}.json"
            )

            save_text(
                result,
                RAW_TEXT_DIR / f"{file_stem}.txt"
            )

            save_tables(
                result,
                RAW_TABLES_DIR / f"{file_stem}_tables.json"
            )

            logger.info(f"Processing completed: {pdf_path.name}")

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
            raise

    def run(self):
        try:
            logger.info("Starting ingestion pipeline")

            pdf_files = get_pdf_files(RAW_PDF_DIR)

            if not pdf_files:
                logger.warning("No PDFs found in raw directory")
                return

            logger.info(f"Found {len(pdf_files)} PDF(s) to process")

            for pdf in pdf_files:
                try:
                    self.process_single_pdf(pdf)
                except Exception as e:
                    logger.error(f"Failed processing file: {pdf.name}", exc_info=True)
                    continue  # continue processing next files

            logger.info("Ingestion pipeline completed")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
