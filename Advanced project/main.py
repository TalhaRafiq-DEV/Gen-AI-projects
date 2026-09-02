from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import ask_question


# --------------------------------
# 1. Create FastAPI application
# --------------------------------

app = FastAPI(
    title="PDF RAG API",
    description="Ask questions about a PDF using RAG",
    version="1.0"
)


# --------------------------------
# 2. Request model
# --------------------------------

class QuestionRequest(BaseModel):

    question: str


# --------------------------------
# 3. Basic test endpoint
# --------------------------------

@app.get("/")
def home():

    return {
        "message": "PDF RAG API is running"
    }


# --------------------------------
# 4. RAG endpoint
# --------------------------------

@app.post("/ask")
def ask(request: QuestionRequest):

    answer = ask_question(
        request.question
    )

    return {
        "question": request.question,
        "answer": answer
    }