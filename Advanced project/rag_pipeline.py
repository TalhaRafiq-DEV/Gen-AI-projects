import os

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


# --------------------------------
# 1. Load API key
# --------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# --------------------------------
# 2. Create Gemini embeddings
# --------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)


# --------------------------------
# 3. Connect to ChromaDB
# --------------------------------

vectorstore = Chroma(
    collection_name="oop_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------
# 4. Create LangChain Retriever
# --------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# --------------------------------
# 5. Create Gemini LLM
# --------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0
)


# --------------------------------
# 6. Create prompt
# --------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions about the provided PDF.

Use ONLY the information provided in the context.

If the answer is not present in the context, say:

"I could not find the answer in the provided PDF."

Context:
{context}

Question:
{question}

Give a clear and simple answer.
"""
)


# --------------------------------
# 7. RAG function
# --------------------------------

def ask_question(question):

    # Retrieve relevant chunks
    documents = retriever.invoke(question)

    # Combine chunks into context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create the final prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    # Send prompt to Gemini
   # response = llm.invoke(final_prompt)
   # return response.content
    # Send prompt to Gemini
    response = llm.invoke(final_prompt)

    if isinstance(response.content, list):
        return "\n".join(
            item["text"]
            for item in response.content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return response.content