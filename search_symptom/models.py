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

class stopword(models.Model):
    stopword = models.TextField()  # 불용어
    
    def __str__(self):
        return f'{self.pk}'

class hospital(models.Model):
    ID = models.CharField(max_length=10, primary_key=True)  # 병원ID
    ADRESS = models.CharField(max_length=100)               # 병원주소
    NAME = models.CharField(max_length=100)                 # 병원명
    KIND = models.CharField(max_length=10)                  # 병원분류명
    DEPT = models.CharField(max_length=100)                 # 진료과
    EMERGENCY = models.BooleanField()                       # 응급실운영여부
    BIGO = models.TextField()                               # 비고
    INFO = models.TextField()                               # 병원설명
    LOCATION = models.CharField(max_length=100)             # 간이약도
    MAIN_TEL = models.CharField(max_length=20)              # 대표전화
    EMER_TEL = models.CharField(max_length=20)              # 응급실전화
    WEEKDAY_TIME = models.CharField(max_length=20)          # 평일운영시간
    SAT_TIME = models.CharField(max_length=20)              # 토요일운영시간
    SUN_TIME = models.CharField(max_length=20)              # 일요일운영시간
    HOLIDAY_TIME = models.CharField(max_length=20)          # 공휴일운영시간
    LON = models.FloatField()                               # 병원경도
    LAT = models.FloatField()                               # 병원위도

    def __str__(self):
        return f'{self.NAME}'