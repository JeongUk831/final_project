import pandas as pd
import pymongo

def select_dept(self, dept) :
    question_list = [    {"name": "소화계"}, {"name": "내분비계"}, {"name": "감각기능계"}, {"name": "근골격계"},    
                        {"name": "감염성"}, {"name": "호흡기계"}, {"name": "심혈관계"},    
                        {"name": "피부질환"}, {"name": "정신신경계"}]

    #소화계 질문 리스트
    question_list_digestion = { 
        "구토": "구토한 적이 있으신가요?",
        "복부 통증": "배가 아파서 불편하신가요?",
        "오심": "토할 것 같은 느낌이 드시나요?",
        "설사": "자주 설사하시나요?",
        "빈혈": "빈혈이 있으신가요?",
        "권태감": "기력이 없거나 무기력한 느낌이 드시나요?",
        "덩어리가 만져짐": "배에 덩어리나 혹이 느껴지시나요?",
        "복부팽만감": "배에 가스가 차서 불편한 느낌이 드시나요?",
        "변비": "배변이 잘 안되시나요?"
    }
    #내분비계 질문 리스트
    question_list_endocrine = {
        "열": "체온이 37.5도 이상인가요?",
        "구토": "구토를 경험하신 적이 있으신가요?",
        "두통": "두통이 있으신가요?",
        "기침": "기침을 자주 하시나요?",
        "식욕부진": "식욕이 없으신가요?",
        "권태감": "무기력하고 힘이 없으신가요?",
        "피로감": "평소에 자주 피곤하시나요?",
    }
    #감각기능계 질문 리스트
    question_list_sense = {
        "시야장애": "시야가 불편하거나 뿌옇게 보이시나요?",
        "압통": "압력이나 눌림으로 인해 통증이 느껴지나요?",
        "눈의 충혈": "눈이 붉거나 충혈되어 보이나요?",
        "청력장애": "청력이 저하되어 소리를 제대로 듣지 못하시나요?",
        "눈부심": "눈이 아프거나 시력이 저하되면서 눈부심이 느껴지나요?",
        "눈의 통증": "눈에서 통증이 느껴지나요?",
        "이명": "귀에서 소리가 들리지만 실제 외부 소리가 아닌 이명 증상이 있나요?"
    }
    #근골격계 질문 리스트
    question_list_musculoskeletal = {
        "근육통": "근육통이 있으십니까?",
        "요통": "허리가 아프십니까?",
        "근력 약화": "몸이 힘들게 느껴지고 쉽게 지치십니까?",
        "관절 운동성 감소": "운동을 할 때 관절이 뻣뻣하거나 아프십니까?",
        "무릎 부위 통증": "무릎이 아프거나 불편하신가요?",
        "어깨의 통증": "어깨가 아프거나 불편하신가요?",
        "뼈의 변형": "뼈가 비정상적으로 변형되거나 뒤틀렸나요?"
    }
    #감염성 질문 리스트
    question_list_infectious = {
    "열": "체온이 높아진 적이 있으시나요?",
    "구토": "구토를 한 적이 있으시나요?",
    "오심": "구토를 하지는 않았지만 토할 것 같은 느낌이 드시나요?",
    "기침": "기침을 자주 하시나요?",
    "설사": "설사를 자주 하시나요?",
    "발진": "피부에 발진이 있으시나요?",
    "오한": "추위를 느끼거나 몸이 떨리는 증상이 있으시나요?",
    "근육통": "근육이 아프거나 통증이 있으시나요?",
    "혈변": "대변이 빨갛게 변한 적이 있으시나요?",
    "코막힘": "코가 막혀 숨쉬기 어려움이 있으시나요?"
    }
    #호흡기계 질문 리스트
    question_list_respiratory = {
        "호흡곤란": "호흡곤란 증상이 있습니까?",
        "기침": "기침을 자주 하십니까?",
        "삼키기 곤란": "식사를 할 때 삼키기 힘드십니까?",
        "가래": "가래가 자주 나오십니까?",
        "운동 시 호흡곤란": "운동을 할 때 숨쉬기 힘드십니까?",
    }
    #심혈관계 질문 리스트
    question_list_cardiovascular = {
    "가슴 통증": "가슴 통증이나 압박감이 있으신가요?",
    "황달":"피부와 눈이 노랗게 변하신 적이 있으신가요?",
    "빈혈":"지친 느낌이나 쉽게 피로감을 느끼시나요?",
    "저혈압":"어지러움이나 혹시 좌식에서 일어날 때 기절하신 적이 있으신가요?",
    "고혈압":"혈압이 평소보다 높은 것 같은 느낌이 있으신가요? ",
    "위장관 출혈":"복부 통증, 설사, 혹은 토혈 등이 있으신가요?",
    "뇌전증 발작":"뇌전증 발작이나 경험적으로 뇌에 이상이 있으신 적이 있으신가요?"
    }
    #피부질환 질문 리스트
    question_list_skin = {
    "발진": "발진이 있으십니까?",
    "피부소양감": "피부에 가려움증, 따갑거림 등의 소양감을 느끼시나요?",
    "물집": "피부에 물집이 생겼나요?",
    "피부 긴장도 저하": "피부가 늘어져 보이거나 탄력이 없어진 느낌이 드시나요?",
    "소양감": "피부에 가려움증, 따갑거림 등의 소양감이 있으신가요?",
    "빈맥": "피부에 정맥이 부어보이거나 구부러진 느낌이 드시나요?",
    "환부의 분비물": "피부에 분비물이 나오거나 따가운 감촉이 있으신가요?",
    "피부 건조": "피부가 건조한 느낌이 드시나요?"
    }
    #정신신경계 질문 리스트
    question_list_neuro = {
    "두통": "두통이 있으십니까?",
    "어지러움": "어지러움이나 현기증이 있으십니까?",
    "시야장애": "시야장애가 있으십니까?",
    "이명": "이명증상이 있으십니까?",
    "불안": "불안이나 조현증 증상이 있으신가요?",
    "의식 변화": "의식 변화가 있으십니까?",
    "뇌전증 발작": "뇌전증 발작이나 경련 증상이 있으신가요?"
    }
    ################################################################################################
    # 질문에 대한 답변을 저장할 딕셔너리
    symptom_dict_digestion = {
        "구토": "구토",
        "복부 통증": "복부 통증",
        "오심": "오심",
        "설사": "설사",
        "빈혈": "빈혈",
        "권태감": "권태감",
        "덩어리가 만져짐": "덩어리가 만져짐",
        "복부팽만감": "복부팽만감"
    }

    symptom_dict_endocrine = {
        "열": "열",
        "구토": "구토",
        "두통": "두통",
        "기침": "기침",
        "식욕부진": "식욕부진",
        "권태감": "권태감",
        "피로감": "피로감"
    }

    symptom_dict_sense = {
        "시야장애": "시야장애",
        "압통": "압통",
        "눈의 충혈": "눈의 충혈",
        "청력장애": "청력장애",
        "눈부심": "눈부심",
        "눈의 통증": "눈의 통증",
        "이명": "이명"
    }

    symptom_dict_musculoskeletal = {
        "근육통": "근육통",
        "요통":"요통",
        "근력 약화": "근력 약화",
        "관절 운동성 감소": "관절 운동성 감소",
        "무릎 부위 통증": "무릎 부위 통증",
        "어깨의 통증": "어깨의 통증",
        "뼈의 변형": "뼈의 변형"
        
    }

    symptom_dict_infectious = {
        "열": "열",
        "오심": "오심",
        "구토": "구토",
        "기침": "기침",
        "설사": "설사",
        "발진": "발진",
        "오한": "오한",
        "근육통": "근육통",
        "혈변": "혈변",
        "코막힘": "코막힘"
    }

    symptom_dict_respiratory = {
    "호흡곤란": "호흡곤란",
    "기침": "기침",
    "삼키기 곤란": "삼키기 곤란",
    "가래": "가래",
    "운동 시 호흡곤란": "운동 시 호흡곤란"
    }

    symptom_dict_cardiovascular = {
        "가슴 통증": "가슴 통증",
        "황달": "황달",
        "빈혈": "빈혈",
        "저혈압": "저혈압",
        "고혈압": "고혈압",
        "위장관 출혈": "위장관 출혈",
        "뇌전증 발작": "뇌전증 발작"
    }

    symptom_dict_skin = {
    "발진": "발진",
    "피부소양감": "피부소양감",
    "물집": "물집?",
    "피부 긴장도 저하": "피부 긴장도 저하",
    "소양감": "소양감",
    "빈맥": "빈맥",
    "환부의 분비물": "환부의 분비물",
    "피부 건조": "피부 건조"
    }

    symptom_dict_neuro = {
    "두통": "두통",
    "어지러움": "어지러움",
    "시야장애": "시야장애",
    "이명": "이명증상",
    "불안": "불안",
    "의식 변화": "의식 변화",
    "뇌전증 발작": "뇌전증 발작"
    }

    # 질문에 대한 답변을 저장할 딕셔너리
    answer_dict = {}
    answer_dict_j = {}

    # CSV 파일에서 데이터를 읽어옴
    df = pd.read_csv("disease.csv", encoding='cp949')
    # MongoDB와 연결 수정 (가능하면)

    #체크박스에 따라 if 문 생성
    if dept=='소화계':  # 소화계 체크박스가 선택되어 있는 경우
        for key, question in question_list_digestion.items():
            answer = input(question)
            answer_dict_j[key] = answer.upper() == "YES"
        filtered_df = df.copy()
        for key, value in answer_dict_j.items():
            if value:
                symptom = symptom_dict_digestion[key]
                filtered_df = filtered_df[filtered_df["증상키워드"].str.contains(symptom)]
                
    elif dept=='내분비계':
        for key, question in question_list_endocrine.items():
            answer = input(question)
            answer_dict_j[key] = answer.upper() == "YES"
        filtered_df = df.copy()
        for key, value in answer_dict_j.items():
            if value:
                symptom = symptom_dict_endocrine[key]
                filtered_df = filtered_df[filtered_df["증상키워드"].str.contains(symptom)]
                
    elif dept=='감각기능계':
        for key, question in question_list_sense.items():
            answer = input(question)
            answer_dict_j[key] = answer.upper() == "YES"
        filtered_df = df.copy()
        for key, value in answer_dict_j.items():
            if value:
                symptom = symptom_dict_sense[key]
                filtered_df = filtered_df[filtered_df["증상키워드"].str.contains(symptom)]
                
    elif dept=='근골격계':
        for key, question in question_list_musculoskeletal.items():
            answer = input(question)
            answer_dict_j[key] = answer.upper() == "YES"
        filtered_df = df.copy()
        for key, value in answer_dict_j.items():
            if value:
                symptom = symptom_dict_musculoskeletal[key]
                filtered_df = filtered_df[filtered_df["증상키워드"].str.contains(symptom)]
                
    elif dept=='감염성':
        for key, question in question_list_infectious.items():
            answer = input(question)
            answer_dict_j[key] = answer.upper() == "YES"
        filtered_df = df.copy()
        for key, value in answer_dict_j.items():
            if value:
                symptom = symptom_dict_infectious[key]
                filtered_df = filtered_df[filtered_df["증상키워드"].str.contains(symptom)]
    
    elif dept=='호흡기계':
        for key, question in question_list_respiratory.items():
            answer = input(question)
            answer_dict_j[key] = answer.upper() == "YES"
        filtered_df = df.copy()
        for key, value in answer_dict_j.items():
            if value:
                symptom = symptom_dict_respiratory[key]
                filtered_df = filtered_df[filtered_df["증상키워드"].str.contains(symptom)]
    
    elif dept=='심혈관계':
        for key, question in question_list_cardiovascular.items():
            answer = input(question)
            answer_dict_j[key] = answer.upper() == "YES"
        filtered_df = df.copy()
        for key, value in answer_dict_j.items():
            if value:
                symptom = symptom_dict_cardiovascular[key]
                filtered_df = filtered_df[filtered_df["증상키워드"].str.contains(symptom)]
    
    elif dept=='피부질환':
        for key, question in question_list_skin.items():
            answer = input(question)
            answer_dict_j[key] = answer.upper() == "YES"
        filtered_df = df.copy()
        for key, value in answer_dict_j.items():
            if value:
                symptom = symptom_dict_skin[key]
                filtered_df = filtered_df[filtered_df["증상키워드"].str.contains(symptom)]

    elif dept=='정신신경계':
        for key, question in question_list_neuro.items():
            answer = input(question)
            answer_dict_j[key] = answer.upper() == "YES"
        filtered_df = df.copy()
        for key, value in answer_dict_j.items():
            if value:
                symptom = symptom_dict_neuro[key]
                filtered_df = filtered_df[filtered_df["증상키워드"].str.contains(symptom)]
                
    #체크박스에 따라 if 문 생성            
    for question in question_list:
        answer_dict[question["name"]] = dept
    
    print(answer_dict)
