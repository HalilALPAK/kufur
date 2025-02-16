import json
import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from youtube_transcript_api import YouTubeTranscriptApi
from torch.utils.data import Dataset

# Türkçe Stop Words Listesi
turkish_stop_words = ["ve", "bir", "bu", "o", "ama", "ne", "de", "da", "için", "gibi", "ile", "en", "daha", "çok"]

# Küfür listesi yükleme
with open("swear.json", "r", encoding="utf-8") as file:
    kufur_data = json.load(file)
kufur_listesi = kufur_data.get("tr", [])

# Leetspeak dönüşümleri
def leetspeak_normalize(text):
    replacements = {"@": "a", "$": "s", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s"}
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

# Önişleme fonksiyonu
def preprocess_text(text):
    text = text.lower()
    text = leetspeak_normalize(text)
    text = re.sub(r'[^a-zçğıöşü0-9 ]+', '', text)
    words = text.split()
    words = [word for word in words if word not in turkish_stop_words]
    return " ".join(words)

# Küfürlü ve temiz cümleler oluşturma
kufur_cumleler = [f"Sen bir {kufur}sin!" for kufur in kufur_listesi]
temiz_cumleler = ["Bugün hava çok güzel.", "İnsanlarla güzel ilişkiler kuruyorum.", "Çalışmalarıma odaklanıyorum."] * 2

data = {"text": kufur_cumleler + temiz_cumleler, "label": [1] * len(kufur_cumleler) + [0] * len(temiz_cumleler)}
df = pd.DataFrame(data)
df["text"] = df["text"].apply(preprocess_text)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Eğitim-test bölme
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# BERT tabanlı model yükleme
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-uncased", num_labels=2)

# Özel veri kümesi sınıfı
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = TextDataset(X_train.tolist(), y_train.tolist())
test_dataset = TextDataset(X_test.tolist(), y_test.tolist())

# Eğitim argümanları
testing_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # FP16 kullanımı ile hız optimizasyonu
    load_best_model_at_end=True,  # En iyi modeli sonunda yükleme
)

# Model eğitme
trainer = Trainer(
    model=model,
    args=testing_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)
trainer.train()

# Modeli kaydet
model.save_pretrained("swear_model")
tokenizer.save_pretrained("swear_model")

# Modeli yükleme
model = AutoModelForSequenceClassification.from_pretrained("swear_model")
tokenizer = AutoTokenizer.from_pretrained("swear_model")

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()

# YouTube video ID
video_id = 'u2Noc_gEZGk'
try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr'])
    for entry in transcript:
        text = preprocess_text(entry['text'])
        score = predict_text(text)
        if score > 0.7:
            print(f"Küfür Tespit Edildi ({score:.2f}): {text}")
        else:
            print(f"Güven Skoru ({score:.2f}): {text}")
except Exception as e:
    print(f"YouTube altyazısı alınamadı: {e}")

from transformers import BertForSequenceClassification

# Modeli kaydet
model.save('model.h5')  # veya model.save('model.keras') kullanabilirsiniz


from transformers import BertTokenizer

# Tokenizer'ı kaydetme
tokenizer.save_pretrained("path_to_save_model")