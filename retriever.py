import os

import chromadb
from dotenv import load_dotenv
from google import genai


# -------------------------
# 1. Load API key
# -------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# -------------------------
# 2. Connect to ChromaDB
# -------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# -------------------------
# 3. Get our collection
# -------------------------

collection = chroma_client.get_collection(
    name="oop_documents"
)


# -------------------------
# 4. Ask a question
# -------------------------

question = input("Ask a question about the PDF: ")


# -------------------------
# 5. Convert question to vector
# -------------------------

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

query_embedding = result.embeddings[0].values


# -------------------------
# 6. Search ChromaDB
# -------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


# -------------------------
# 7. Display retrieved chunks
# -------------------------

print("\n===== RETRIEVED CHUNKS =====")

documents = results["documents"][0]

for i, document in enumerate(documents):
    print(f"\n----- RESULT {i + 1} -----")
    print(document)