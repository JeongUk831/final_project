from djongo import models

# Create your models here.
class hospital(models.Model):
    ID = models.CharField(max_length=10, primary_key=True)  # 병원ID
    ADRESS = models.CharField(max_length=100)               # 병원주소
    KIND = models.CharField(max_length=10)                  # 병원분류명
    EMERGENCY = models.BooleanField()                       # 응급실운영여부
    BIGO = models.TextField()                               # 비고
    INFO = models.TextField()                               # 병원설명
    LOCATION = models.CharField(max_length=100)             # 간이약도
    NAME = models.CharField(max_length=100)                 # 병원명
    MAIN_TEL = models.CharField(max_length=20)              # 대표전화
    EMER_TEL = models.CharField(max_length=20)              # 응급실전화
    WEEKDAY_TIME = models.CharField(max_length=20)          # 평일운영시간
    SAT_TIME = models.CharField(max_length=20)              # 토요일운영시간
    SUN_TIME = models.CharField(max_length=20)              # 일요일운영시간
    HOLIDAY_TIME = models.CharField(max_length=20)          # 공휴일운영시간
    LON = models.FloatField()                               # 병원경도
    LAT = models.FloatField()                               # 병원위도
    DEPT = models.CharField(max_length=100)                 # 진료과

    class Meta:
        managed = False
        db_table = "hospital"  # MongoDB Collection 이름

    #def __str__(self):
    #    return f'{self.NAME}'