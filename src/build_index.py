from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

pdf_folder = "data/pdfs"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_chunks = []       # text content of each chunk
all_metadata = []     # which file + chunk number each came from

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_folder, filename)
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({"source": filename, "chunk_id": i})

print(f"Total chunks created: {len(all_chunks)}")

# Load a local embedding model (downloads once, ~420MB)
print("Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")

print("Generating embeddings (this may take a few minutes)...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save everything
os.makedirs("data/index", exist_ok=True)
faiss.write_index(index, "data/index/faiss.index")

with open("data/index/chunks.pkl", "wb") as f:
    pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)

print("Done. Index and chunks saved to data/index/")