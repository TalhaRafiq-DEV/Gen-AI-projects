from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Read PDF

pdf_path = "documents/OOP.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text


# 2. Create LangChain text splitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# 3. Split the PDF text

chunks = splitter.split_text(full_text)


# 4. Check the result

print("Total characters:", len(full_text))
print("Total chunks:", len(chunks))


for i, chunk in enumerate(chunks):
    print(f"\n----- CHUNK {i + 1} -----")
    print(chunk)