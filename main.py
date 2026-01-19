from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings #type: ignore
from langchain_core.documents import Document # Yeni ve güvenli yol
import json
import os

app = FastAPI()

# WordPress bağlantısı için güvenlik izinleri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Embeddings: Metinleri sayısallaştıran modern kütüphane
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-miniLM-L6-v2")

# 2. Veri Hazırlama ve FAISS Dizini Oluşturma
def prepare_faiss_index():
    jsonl_path = "webreta.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"HATA: {jsonl_path} dosyası bulunamadı!")
        return None
    
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # RAG için içerik ve başlığı birleştiriyoruz
            content = f"Soru: {data['instruction']}\nYanıt: {data['response']}"
            documents.append(Document(page_content=content))
    
    # FAISS vektör deposunu oluştur
    return FAISS.from_documents(documents, embeddings)

vector_db = prepare_faiss_index()

# İndirdiğin Llama 3.2 modelini çağır
llm = Ollama(model="llama3.2")

@app.get("/ask")
async def ask_webreta(question: str):
    if vector_db is None:
        return {"response": "Sistem veritabanı yüklenemedi."}
    
    # RAG: Soruna en yakın bilgiyi FAISS ile bul
    docs = vector_db.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    # AI System Prompt
    prompt = f"""Sen Webreta firmasının asistanısın. 
    Sadece aşağıdaki bilgileri kullanarak Türkçe cevap ver. 
    Bilgin dışındaysa 'Bu konuda bilgi veremem' de.
    
    BİLGİ:
    {context}
    
    SORU: {question}
    """
    
    try:
        response = llm.invoke(prompt)
        return {"response": response}
    except Exception as e:
        return {"response": f"Hata oluştu: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print("Webreta AI Sunucusu Başlatılıyor...")
    uvicorn.run(app, host="0.0.0.0", port=8000)