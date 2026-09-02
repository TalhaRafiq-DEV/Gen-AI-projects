import os

import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------
# 1. Load API key
# --------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# --------------------------------
# 2. Read PDF
# --------------------------------

pdf_path = "documents/OOP.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text


# --------------------------------
# 3. Split text into chunks
# --------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(full_text)

print("Total chunks:", len(chunks))


# --------------------------------
# 4. Create embeddings
# --------------------------------

embeddings = []

for chunk in chunks:

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )

    vector = result.embeddings[0].values

    embeddings.append(vector)


print("Total embeddings:", len(embeddings))


# --------------------------------
# 5. Create ChromaDB
# --------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# --------------------------------
# 6. Create collection
# --------------------------------

collection = chroma_client.get_or_create_collection(
    name="oop_documents"
)


# --------------------------------
# 7. Store chunks + vectors
# --------------------------------

ids = []

for i in range(len(chunks)):
    ids.append(str(i))


collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks
)


print("Data stored in ChromaDB!")


# --------------------------------
# 8. Check stored data
# --------------------------------

data = collection.get()

print("Number of documents in DB:", len(data["documents"]))

for i, document in enumerate(data["documents"]):
    print(f"\n----- STORED DOCUMENT {i + 1} -----")
    print(document)

    data = collection.get()

print("Number of documents:", len(data["documents"]))
print("Number of IDs:", len(data["ids"]))

for i in range(len(data["documents"])):
    print(f"\n----- DOCUMENT {i + 1} -----")
    print("ID:", data["ids"][i])
    print("Text:", data["documents"][i][:100])