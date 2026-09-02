from pypdf import PdfReader


pdf_path = "documents/OOP.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text


chunk_size = 500
chunk_overlap = 100

chunks = []

step = chunk_size - chunk_overlap

for i in range(0, len(full_text), step):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)


print("Total characters:", len(full_text))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n----- CHUNK {i + 1} -----")
    print(chunk)