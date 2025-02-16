from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("model.h5") 

class VideoRequest(BaseModel):
    video_id: str

@app.post("/predict")
async def predict(request: VideoRequest):
    video_id = request.video_id
    
    # Video metnini almak için API çağrısı yapın (YouTube API veya başka bir kaynak)
    video_text = get_video_text(video_id)  # Burada video metnini almak için bir işlev çağırıyoruz
    
    # Modelinize metni gönderip küfür tespiti yapın
    prediction = model.predict([video_text])
    
    # Küfürlü kelime tespit edilirse, yanıtı dönün
    return {"video_id": video_id, "contains_swear": prediction[0] > 0.5}

def get_video_text(video_id: str):
    # Video ID'sine bağlı açıklama veya metinleri almak için buraya YouTube API çağrısı ekleyin
    # Örnek: YouTube API kullanarak açıklama almak
    return "This is a sample description of the video."

