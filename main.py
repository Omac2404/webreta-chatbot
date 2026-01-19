from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
import json
import os

app = FastAPI()

# WordPress ve dış dünya bağlantısı için CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render üzerinden gelen API anahtarını al
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Groq üzerinde şu an en güncel ve kararlı model: llama-3.3-70b-versatile
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", 
    api_key=GROQ_API_KEY
)

def get_context(question: str):
    """Bellek harcamadan JSONL dosyasında esnek arama yapar."""
    context = ""
    question_lower = question.lower()
    
    try:
        if not os.path.exists("webreta.jsonl"):
            return "Webreta, İzmir Bornova merkezli profesyonel bir dijital reklam ajansıdır."

        with open("webreta.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                instr = data.get('instruction', '').lower()
                resp = data.get('response', '')
                
                # Soru içindeki anahtar kelimeleri dosyada ara
                keywords = [kw for kw in question_lower.split() if len(kw) > 2]
                if any(kw in instr for kw in keywords):
                    context += f"- {resp}\n"
        
        # Eşleşme yoksa dosyanın başındaki genel bilgileri kullan
        if not context and len(lines) > 0:
            context = json.loads(lines[0]).get('response', '')

    except Exception as e:
        print(f"Veri okuma hatası: {e}")
    
    return context[:1500]

@app.get("/ask")
async def ask_webreta(question: str):
    # Render loglarında gelen soruyu görmemizi sağlar
    print(f"Gelen Soru: {question}")
    
    context = get_context(question)
    
    prompt = f"""Sen Webreta Dijital Reklam Ajansı'nın uzman asistanısın. 
    Sana sunulan kurumsal bilgileri kullanarak müşteriye yardımcı ol.
    Yanıtların profesyonel, samimi ve kısa olsun. 
    Bilmediğin konularda ajansın iletişim bilgilerini ver.

    KURUMSAL BİLGİLER:
    {context}
    
    MÜŞTERİ SORUSU: {question}
    YANIT:"""
    
    try:
        response = llm.invoke(prompt)
        return {"response": response.content}
    except Exception as e:
        # Hata detayını Render loglarına basar
        print(f"Groq API Hatası: {str(e)}")
        return {"response": "Şu an bağlantıda bir sorun var, lütfen 30 saniye sonra tekrar deneyin."}

if __name__ == "__main__":
    import uvicorn
    # Render'ın dinamik port atamasıyla uyumlu çalışır
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
