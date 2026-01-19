from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
import json
import os

app = FastAPI()

# WordPress bağlantısı için CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Anahtarı kontrolü
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("KRİTİK HATA: GROQ_API_KEY ortam değişkeni ayarlanmamış!")

llm = ChatGroq(
    temperature=0, 
    model_name="llama3-8b-8192", 
    api_key=GROQ_API_KEY
)

def get_context(question: str):
    """RAM harcamadan JSONL dosyasında anahtar kelime araması yapar."""
    context = ""
    question_lower = question.lower()
    
    try:
        if not os.path.exists("webreta.jsonl"):
            print("HATA: webreta.jsonl dosyası bulunamadı!")
            return "Webreta, İzmir Bornova merkezli bir dijital reklam ajansıdır."

        with open("webreta.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Dosyada soruya en yakın anahtar kelimeleri ara
            for line in lines:
                data = json.loads(line)
                instr = data.get('instruction', '').lower()
                resp = data.get('response', '')
                
                # Soru içindeki kelimeleri (3 harften büyük) dosyada ara
                keywords = [kw for kw in question_lower.split() if len(kw) > 2]
                if any(kw in instr for kw in keywords):
                    context += f"- {resp}\n"
        
        # Eğer spesifik bir bilgi bulunamazsa dosyanın en başındaki genel bilgileri ver
        if not context and len(lines) > 0:
            context = json.loads(lines[0]).get('response', '')

    except Exception as e:
        print(f"Arama işlemi sırasında hata: {e}")
    
    return context[:1500] # Karakter sınırını koru

@app.get("/ask")
async def ask_webreta(question: str):
    # Log: Gelen soruyu Render terminalinde gör
    print(f"Gelen Soru: {question}")
    
    context = get_context(question)
    
    prompt = f"""Sen Webreta dijital reklam ajansının resmi asistanısın. 
    Sadece sana verilen aşağıdaki kurumsal bilgileri kullanarak cevap ver. 
    Bilmediğin bir konu olursa nazikçe ajansla iletişime geçmelerini söyle.

    KURUMSAL BİLGİLER:
    {context}
    
    MÜŞTERİ SORUSU: {question}
    YANIT:"""
    
    try:
        response = llm.invoke(prompt)
        return {"response": response.content}
    except Exception as e:
        print(f"Groq API Hatası: {e}")
        return {"response": "Şu an yanıt sisteminde bir yoğunluk var, lütfen birazdan tekrar deneyin."}

if __name__ == "__main__":
    import uvicorn
    # Render'ın dinamik port atamasına uyum sağla
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
