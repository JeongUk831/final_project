from django.contrib import admin
from search_symptom.models import disease, hospital

# Register your models here.
admin.site.register(disease)
admin.site.register(hospital)