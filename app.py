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