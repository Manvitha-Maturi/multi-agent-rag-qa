from PyPDF2 import PdfReader
import os

#Defines the folder where you want to extract text from PDF files. 
pdf_folder = "data/pdfs"

#Loops through all files in the specified folder
for filename in os.listdir(pdf_folder):
     #Makes sure to only process files that end with the .pdf extension
     if filename.endswith(".pdf"):
        #Combines the folder path and filename to get the full path to the PDF file
        path = os.path.join(pdf_folder, filename)

        #Load the PDF file into PyPDF's reader and extract text from each page. 
        reader = PdfReader(path)
        text = ""

        # Loop through every page in this specific PDF and extract its text
        for page in reader.pages:

            # If extract_text() returns None (empty page), default to an empty string ""
            text += page.extract_text() or ""

        # Print the results for this PDF file
        print(f"--- {filename} ---")
        print(f"Pages: {len(reader.pages)}")
        print(f"First 300 chars: {text[:300]}")
        print()