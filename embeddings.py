import os

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------
# 1. Load API key
# -------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# -------------------------
# 2. Read PDF
# -------------------------

pdf_path = "documents/OOP.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text


# -------------------------
# 3. Create chunks
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(full_text)

print("Total chunks:", len(chunks))


# -------------------------
# 4. Create embeddings
# -------------------------

embeddings = []

for chunk in chunks:

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )

    vector = result.embeddings[0].values

    embeddings.append(vector)


# -------------------------
# 5. Check result
# -------------------------

print("Total embeddings:", len(embeddings))

print("Vector size:", len(embeddings[0]))

print("First 10 values of first vector:")
print(embeddings[0][:10])