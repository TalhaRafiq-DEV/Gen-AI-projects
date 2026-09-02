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
# 3. Get collection
# -------------------------

collection = chroma_client.get_collection(
    name="oop_documents"
)


# -------------------------
# 4. Get user's question
# -------------------------

question = input("Ask a question about the PDF: ")


# -------------------------
# 5. Convert question to embedding
# -------------------------

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

query_embedding = result.embeddings[0].values


# -------------------------
# 6. Search Vector DB
# -------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

documents = results["documents"][0]


# -------------------------
# 7. Combine retrieved chunks
# -------------------------

context = "\n\n".join(documents)


# -------------------------
# 8. Create prompt
# -------------------------

prompt = f"""
You are a helpful assistant answering questions about the provided PDF.

Use ONLY the information in the context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided PDF."

Context:
{context}

Question:
{question}

Give a clear and simple answer.
"""


# -------------------------
# 9. Generate answer
# -------------------------

print("\nGenerating answer...")

try:

    response = client.models.generate_content(
        model="gemini-3.7-flash",
        contents=prompt
    )

    print("\n===== ANSWER =====")
    print(response.text)

except Exception as e:

    print("\nGemini generation failed.")
    print("Error:", e)