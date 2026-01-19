from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Anahtarını Render'dan alacağız
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-miniLM-L6-v2")

def prepare_faiss_index():
    jsonl_path = "webreta.jsonl"
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            content = f"Soru: {data['instruction']}\nYanıt: {data['response']}"
            documents.append(Document(page_content=content))
    return FAISS.from_documents(documents, embeddings)

vector_db = prepare_faiss_index()

@app.get("/ask")
async def ask_webreta(question: str):
    docs = vector_db.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"Sen Webreta asistanısın. Sadece şu bilgiyi kullan: {context}\nSoru: {question}"
    
    try:
        response = llm.invoke(prompt)
        return {"response": response.content}
    except Exception as e:
        return {"response": "Bağlantı hatası oluştu."}

if __name__ == "__main__":
    import uvicorn
    # Render'ın verdiği PORT numarasını kullan, yoksa 8000'i kullan
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)

