from PyPDF2 import PdfReader
import os

pdf_folder = "data/pdfs"

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_folder, filename)
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        print(f"--- {filename} ---")
        print(f"Pages: {len(reader.pages)}")
        print(f"First 300 chars: {text[:300]}")
        print()