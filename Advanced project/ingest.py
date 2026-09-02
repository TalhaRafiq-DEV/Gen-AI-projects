import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma


# --------------------------------
# 1. Load environment variables
# --------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# --------------------------------
# 2. Load PDF
# --------------------------------

pdf_path = "documents/OOP.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("PDF loaded")
print("Total pages:", len(documents))


# --------------------------------
# 3. Split PDF into chunks
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# --------------------------------
# 4. Create Gemini embeddings
# --------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)


# --------------------------------
# 5. Store embeddings in ChromaDB
# --------------------------------

vectorstore = Chroma(
    collection_name="oop_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------
# 6. Add chunks to ChromaDB
# --------------------------------

vectorstore.add_documents(chunks)


print("Documents successfully stored in ChromaDB")