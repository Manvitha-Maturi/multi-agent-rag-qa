# Import the tools we need
from sentence_transformers import SentenceTransformer  # For turning text into numbers
import faiss  # The fast vector search engine
import pickle  # For loading our original text chunks and file names

# 1. Load the exact same AI model we used to build the index.
# This ensures our new question is translated into numbers the exact same way.
model = SentenceTransformer("all-mpnet-base-v2")

# 2. Load the searchable FAISS index file we saved in Step 2.
# (This file contains only the list of numbers/vectors, not the actual text).
index = faiss.read_index("data/index/faiss.index")

# 3. Load the companion file that holds the actual text sentences and PDF file names.
with open("data/index/chunks.pkl", "rb") as f:
    data = pickle.load(f)

# 4. Define your question and use the model to translate it into a list of numbers.
query = "What causes capacity fade in lithium-sulfur batteries?"
query_vec = model.encode([query])  # This outputs a numerical vector representing the meaning

# 5. Tell FAISS how many matching chunks you want to bring back.
k = 3  # We want the top 3 closest matches

# 6. Search the index!
# We pass it the question's numbers (query_vec) and the number of results we want (k).
# It returns:
#   - distances: how close the matches are (smaller number = better match)
#   - indices: the ID numbers/positions of the winning chunks
distances, indices = index.search(query_vec, k)

# 7. Loop through the 3 winning positions and print out their text.
for rank, idx in enumerate(indices[0]):
    print(f"\n--- Result {rank+1} (distance: {distances[0][rank]:.4f}) ---")
    
    # Use the ID number (idx) to look up the original PDF name in our pickle data
    print(f"Source: {data['metadata'][idx]['source']}")
    
    # Use the same ID number to pull the text chunk, cutting it off at 300 characters for readability
    print(f"Text: {data['chunks'][idx][:300]}")