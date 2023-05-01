from django.db import models

# Create your models here.
class disease(models.Model):
    NAME = models.CharField(max_length=100)  # 질병명
    SYMPTOM = models.TextField()             # 증상키워드
    RELATE = models.TextField()              # 관련질환
    DEFINE = models.TextField()              # 정의
    REASON = models.TextField()              # 원인
    SYMPTOM_TEXT = models.TextField()        # 증상설명
    DIAGNOSIS = models.TextField()           # 진단
    THERAPY = models.TextField()             # 치료
    Keyword = models.TextField()             # 키워드
    Token_keyword = models.TextField()       # 토큰키워드
    DEPT = models.CharField(max_length=100)  # 진료과
    Dump_name = models.CharField(max_length=10)  # 군집명
    
    def __str__(self):
        return f'{self.pk}.{self.NAME}'