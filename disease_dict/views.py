from django.shortcuts import render
from disease_dict.models import disease
# Create your views here.
def disease_dict(request):
    diseases = disease.objects.all()  # DB에서 모든 disease의 레코드를 가져온다.
    query = request.GET.get('q')  # 검색어 가져오기
    if not query:
        results = "질병을 입력하세요."
        return render(request, 'disease_dict/disease_dict.html', {'results':results})  # 질병명을 검색하지 않은 경우

    diseases = disease.objects.filter(NAME__icontains = query)  # 질병명 검색
    if not diseases:
        results = "결과가 없습니다."
        return render(request, 'disease_dict/disease_dict.html', {'results':results})

    return render(
        request,
        'disease_dict/disease_dict.html',
        {'diseases':diseases}
    )