import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# KoBERT 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
model = AutoModelForSequenceClassification.from_pretrained("monologg/kobert")

# CSV 파일에서 질병과 증상 매핑 데이터셋 불러오기
with open('disease_symptom_mapping.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]  # 첫 번째 줄은 헤더이므로 제외

# 증상 입력 받기
symptoms = input("증상을 입력하세요: ")

# 증상을 KoBERT 입력 형식에 맞게 변환
inputs = tokenizer(symptoms, return_tensors='pt')

# 모델에 입력하여 예측 결과 출력
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
predicted_disease = lines[predictions.item()].split(',')[0]

print(predictions.item())
print("예측된 질병:", predicted_disease)