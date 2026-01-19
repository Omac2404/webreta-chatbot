from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Anahtarı
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=GROQ_API_KEY)

# RAM Dostu Veri Yükleme (Vektör Veritabanı Yerine Basit Arama)
def get_context(question: str):
    context = ""
    try:
        with open("webreta.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Basit kelime eşleşmesi ile RAM harcamadan arama yapar
                if any(word.lower() in data['instruction'].lower() for word in question.split()):
                    context += f"Soru: {data['instruction']}\nYanıt: {data['response']}\n\n"
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
    return context[:2000] # Çok uzun metinleri kırp

@app.get("/ask")
async def ask_webreta(question: str):
    context = get_context(question)
    
    prompt = f"""Sen Webreta asistanısın. 
    Aşağıdaki bilgileri kullanarak soruyu yanıtla:
    {context}
    
    Soru: {question}"""
    
    try:
        response = llm.invoke(prompt)
        return {"response": response.content}
    except Exception:
        return {"response": "Şu an yanıt veremiyorum."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
