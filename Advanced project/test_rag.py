from rag_pipeline import ask_question


question = input("Ask a question about the PDF: ")

answer = ask_question(question)

print("\n===== ANSWER =====")
print(answer)