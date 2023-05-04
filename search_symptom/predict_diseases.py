import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
#import multiprocessing
from konlpy.tag import Okt
import numpy as np
from imblearn.over_sampling import SMOTE

def predict_diseases(question) :
    result = []
    # 데이터셋 로드
    new_disease = pd.read_csv('./data/new_disease.csv')
    stopword = pd.read_csv('./data/stopword.csv', encoding="utf-8-sig")

    # CountVectorizer와 TfidfTransformer를 사용하여 증상 벡터화
    cv = CountVectorizer()
    tfidf = TfidfTransformer()

    symptoms = new_disease['토큰키워드'].values
    symptom_vectors = cv.fit_transform(symptoms)
    symptom_tfidf = tfidf.fit_transform(symptom_vectors)

    # 클래스 불균형 해결 - SMOTE
    smote = SMOTE(random_state=0, k_neighbors=10)
    symptom_tfidf_over, y_over = smote.fit_resample(symptom_tfidf, new_disease['진료과'])

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(symptom_tfidf_over, y_over, test_size=0.2, random_state=0, stratify=y_over)

    # SVM 모델 학습 - 진료과 예측
    svm = SVC(C=5, gamma='scale', kernel='sigmoid', class_weight='balanced', decision_function_shape='ovo', probability=True, random_state=0)

    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    recall = recall_score(y_test, pred, average="macro")
    precision = precision_score(y_test, pred, average="macro")
    f1 = f1_score(y_test, pred, average='macro')
    #print("svm의 정확도 : {}\nsvm의 재현율 : {}\nsvm의 정밀도 : {}\nsvm F1 : {}".format(accuracy, recall, precision, f1))

    # # 사용자로부터 증상 입력 받기
    # question = input("증상을 입력하세요 : ")  # 두통이 심하고 열이 나고 설사를 한다. // 피부에 두드러기가 나고 열이 난다. // 어지럽고 이명이 들린다. //

    # 입력된 증상을 토큰화하고 정규화하여 진료과 예측에 사용
    t = Okt()
    q_list = [token for token, pos in t.pos(question, stem=True, norm=True) if pos in ['Noun', 'Adjective']]
    qustion_key_list = [lk for lk in list(set(q_list)) if lk not in stopword]
    #print(qustion_key_list)
    user_vector = cv.transform([' '.join(qustion_key_list)])
    user_tfidf = tfidf.transform(user_vector)

    # SVM 모델을 사용하여 진료과 예측
    predicted_probabilities = svm.predict_proba(user_tfidf)[0]

    # 진료과 추천
    top_6_dept = np.argsort(predicted_probabilities)[::-1][:6]
    # print('\n추천 진료과:')
    for i, index in enumerate(top_6_dept):
        # print(f'{i+1}. {svm.classes_[index]} {predicted_probabilities[index]:.4f}')
        result.append(svm.classes_[index])

    if svm.classes_[top_6_dept][0] == '내과':
        new_medicine = new_disease[new_disease['진료과'].str.contains('내과')]
        
        # CountVectorizer와 TfidfTransformer를 사용하여 증상 벡터화
        cv = CountVectorizer()
        tfidf = TfidfTransformer()

        medicine_symptoms = new_medicine['토큰키워드'].values
        medicine_symptom_vectors = cv.fit_transform(medicine_symptoms)
        medicine_symptom_tfidf = tfidf.fit_transform(medicine_symptom_vectors)

        t = Okt()
        q_list = [token for token, pos in t.pos(question, stem=True, norm=True) if pos in ['Noun', 'Adjective']]
        qustion_key_list = [lk for lk in list(set(q_list)) if lk not in stopword]
        user_vector = cv.transform([' '.join(qustion_key_list)])
        user_tfidf = tfidf.transform(user_vector)

        medicine_lr = LogisticRegression(multi_class='multinomial', penalty=None, solver='lbfgs', C=10, random_state=0)

        # 로지스틱 모델 학습 - 군집명 예측
        medicine_lr.fit(medicine_symptom_tfidf, new_medicine['군집명'])

        # 로지스틱 모델을 사용하여 군집명 예측
        predicted_probabilities = medicine_lr.predict_proba(user_tfidf)[0]

        # 내과 - 예측 군집
        all_indices = np.argsort(predicted_probabilities)[::-1]
        #print('\n내과 내원 시 예상 군집명')
        #for i, index in enumerate(all_indices):
        #    print(f'{i+1}. {medicine_lr.classes_[index]} {predicted_probabilities[index]:.4f}')

        # 군집명 중에서 어떤 질병인지 예측
        # print('\n예상 질병')
        cv2 = CountVectorizer()
        tfidf2 = TfidfTransformer()
        new_medicine_detail = new_medicine.loc[new_medicine['군집명'] == medicine_lr.classes_[all_indices][0]]
        new_medicine_detail_symtoms = new_medicine_detail['토큰키워드'].values
        new_medicine_detail_symtom_vectors = cv2.fit_transform(new_medicine_detail_symtoms)
        new_medicine_detail_symtom_tfidf = tfidf2.fit_transform(new_medicine_detail_symtom_vectors)

        user_vector2 = cv2.transform([' '.join(qustion_key_list)])
        user_tfidf2 = tfidf2.transform(user_vector2)

        # medicine_lr - 내과 군집 예측에 사용한 모델
        medicine_lr.fit(new_medicine_detail_symtom_tfidf, new_medicine_detail['질병명'])
        predicted_probabilities2 = medicine_lr.predict_proba(user_tfidf2)[0]
        all_indices2 = np.argsort(predicted_probabilities2)[::-1]

        for i, index in enumerate(all_indices2):
        #     print(f'{i+1}. {medicine_lr.classes_[index]}')  # {predicted_probabilities2[index]:.4f}
            result.append(medicine_lr.classes_[index])

    elif svm.classes_[top_6_dept][0] == '피부과':
        new_skin = new_disease[new_disease['진료과'].str.contains('피부과')]

        cv = CountVectorizer()
        tfidf = TfidfTransformer()

        skin_symptoms = new_skin['토큰키워드'].values
        skin_symptom_vectors = cv.fit_transform(skin_symptoms)
        skin_symptom_tfidf = tfidf.fit_transform(skin_symptom_vectors)

        t = Okt()
        q_list = [token for token, pos in t.pos(question, stem=True, norm=True) if pos in ['Noun', 'Adjective']]
        qustion_key_list = [lk for lk in list(set(q_list)) if lk not in stopword]
        user_vector = cv.transform([' '.join(qustion_key_list)])
        user_tfidf = tfidf.transform(user_vector)

        skin_nb = MultinomialNB(alpha=0.05, fit_prior=True, force_alpha=True)

        # MultinomialNB 모델 학습 - 군집명 예측
        skin_nb.fit(skin_symptom_tfidf, new_skin['군집명'])

        # MultinomialNB 모델을 사용하여 군집명 예측
        predicted_probabilities = skin_nb.predict_proba(user_tfidf)[0]

        # 피부과 - 예측 군집명
        all_indices = np.argsort(predicted_probabilities)[::-1]
        #print('\n피부과 내원 시 예상 군집명')
        #for i, index in enumerate(all_indices):
        #    print(f'{i+1}. {skin_nb.classes_[index]} {predicted_probabilities[index]:.4f}')

        # 군집명 중에서 어떤 질병인지 예측
        # print('\n예상 질병')
        cv2 = CountVectorizer()
        tfidf2 = TfidfTransformer()
        new_skin_detail = new_skin.loc[new_skin['군집명'] == skin_nb.classes_[all_indices][0]]
        new_skin_detail_symtoms = new_skin_detail['토큰키워드'].values
        new_skin_detail_symtom_vectors = cv2.fit_transform(new_skin_detail_symtoms)
        new_skin_detail_symtom_tfidf = tfidf2.fit_transform(new_skin_detail_symtom_vectors)

        user_vector2 = cv2.transform([' '.join(qustion_key_list)])
        user_tfidf2 = tfidf2.transform(user_vector2)

        # skin_nb - 피부과 군집 예측에 사용한 모델
        skin_nb.fit(new_skin_detail_symtom_tfidf, new_skin_detail['질병명'])
        predicted_probabilities2 = skin_nb.predict_proba(user_tfidf2)[0]
        all_indices2 = np.argsort(predicted_probabilities2)[::-1]

        for i, index in enumerate(all_indices2):
            # print(f'{i+1}. {skin_nb.classes_[index]}')  # {predicted_probabilities2[index]:.4f}
            result.append(skin_nb.classes_[index])

    elif svm.classes_[top_6_dept][0] == '이비인후과':
        new_Otorhinolaryngology = new_disease[new_disease['진료과'].str.contains('이비인후과')]

        cv = CountVectorizer()
        tfidf = TfidfTransformer()

        Otorhinolaryngology_symptoms = new_Otorhinolaryngology['토큰키워드'].values
        Otorhinolaryngology_symptom_vectors = cv.fit_transform(Otorhinolaryngology_symptoms)
        Otorhinolaryngology_symptom_tfidf = tfidf.fit_transform(Otorhinolaryngology_symptom_vectors)

        t = Okt()
        q_list = [token for token, pos in t.pos(question, stem=True, norm=True) if pos in ['Noun', 'Adjective']]
        qustion_key_list = [lk for lk in list(set(q_list)) if lk not in stopword]
        user_vector = cv.transform([' '.join(qustion_key_list)])
        user_tfidf = tfidf.transform(user_vector)

        Otorhinolaryngology_nb = MultinomialNB(alpha=0.01, fit_prior=True, force_alpha=True)

        # 나이브베이즈 모델 학습 - 군집명 예측
        Otorhinolaryngology_nb.fit(Otorhinolaryngology_symptom_tfidf, new_Otorhinolaryngology['군집명'])

        # 나이브베이즈 모델을 사용하여 군집명 예측
        predicted_probabilities = Otorhinolaryngology_nb.predict_proba(user_tfidf)[0]

        # 이비인후과 - 예측 군집명
        all_indices = np.argsort(predicted_probabilities)[::-1]
        #print('\n이비인후과 내원 시 예상 군집명')
        #for i, index in enumerate(all_indices):
        #    print(f'{i+1}. {Otorhinolaryngology_nb.classes_[index]} {predicted_probabilities[index]:.4f}')

        # 군집명 중에서 어떤 질병인지 예측
        # print('\n예상 질병')
        cv2 = CountVectorizer()
        tfidf2 = TfidfTransformer()
        new_Otorhinolaryngology_detail = new_Otorhinolaryngology.loc[new_Otorhinolaryngology['군집명'] == Otorhinolaryngology_nb.classes_[all_indices][0]]
        new_Otorhinolaryngology_detail_symtoms = new_Otorhinolaryngology_detail['토큰키워드'].values
        new_Otorhinolaryngology_detail_symtom_vectors = cv2.fit_transform(new_Otorhinolaryngology_detail_symtoms)
        new_Otorhinolaryngology_detail_symtom_tfidf = tfidf2.fit_transform(new_Otorhinolaryngology_detail_symtom_vectors)

        user_vector2 = cv2.transform([' '.join(qustion_key_list)])
        user_tfidf2 = tfidf2.transform(user_vector2)

        # Otorhinolaryngology_nb - 이비인후과 군집 예측에 사용한 모델
        Otorhinolaryngology_nb.fit(new_Otorhinolaryngology_detail_symtom_tfidf, new_Otorhinolaryngology_detail['질병명'])
        predicted_probabilities2 = Otorhinolaryngology_nb.predict_proba(user_tfidf2)[0]
        all_indices2 = np.argsort(predicted_probabilities2)[::-1]

        for i, index in enumerate(all_indices2):
            # print(f'{i+1}. {Otorhinolaryngology_nb.classes_[index]}')  # {predicted_probabilities2[index]:.4f}
            result.append(Otorhinolaryngology_nb.classes_[index])

    elif svm.classes_[top_6_dept][0] == '정형외과':
        new_Orthopedics = new_disease[new_disease['진료과'].str.contains('정형외과')]

        cv = CountVectorizer()
        tfidf = TfidfTransformer()

        Orthopedics_symptoms = new_Orthopedics['토큰키워드'].values
        Orthopedics_symptom_vectors = cv.fit_transform(Orthopedics_symptoms)
        Orthopedics_symptom_tfidf = tfidf.fit_transform(Orthopedics_symptom_vectors)

        t = Okt()
        q_list = [token for token, pos in t.pos(question, stem=True, norm=True) if pos in ['Noun', 'Adjective']]
        qustion_key_list = [lk for lk in list(set(q_list)) if lk not in stopword]
        user_vector = cv.transform([' '.join(qustion_key_list)])
        user_tfidf = tfidf.transform(user_vector)

        lr = LogisticRegression(multi_class='multinomial', penalty=None, solver='newton-cg', C=1, random_state=0)
        svm = SVC(C=3, gamma='scale', kernel='linear', class_weight='balanced', decision_function_shape='ovo', probability=True, random_state=0)
        nb = MultinomialNB(alpha=0.01, fit_prior=True, force_alpha=True)
        rf = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=0)
        mlp = MLPClassifier(hidden_layer_sizes=(10,10), activation='identity', alpha=0.01, max_iter=100, learning_rate='constant', solver='lbfgs', early_stopping=True, batch_size='auto', warm_start=True, random_state=0)

        Orthopedics_soft_model = VotingClassifier(estimators=[('lr', lr), ('svm', svm), ('nb', nb), ('rf', rf), ('mlp', mlp)], voting="soft")

        # 앙상블 소프트 보팅 모델 학습 - 군집명 예측
        Orthopedics_soft_model.fit(Orthopedics_symptom_tfidf, new_Orthopedics['군집명'])

        # 앙상블 소프트 보팅 모델을 사용하여 군집명 예측
        predicted_probabilities = Orthopedics_soft_model.predict_proba(user_tfidf)[0]

        # 정형외과 - 예측 군집명
        all_indices = np.argsort(predicted_probabilities)[::-1]
        #print('\n정형외과 내원 시 예상 군집명')
        #for i, index in enumerate(all_indices):
        #    print(f'{i+1}. {Orthopedics_soft_model.classes_[index]} {predicted_probabilities[index]:.4f}')

        # 군집명 중에서 어떤 질병인지 예측
        # print('\n예상 질병')
        cv2 = CountVectorizer()
        tfidf2 = TfidfTransformer()
        new_Orthopedics_detail = new_Orthopedics.loc[new_Orthopedics['군집명'] == Orthopedics_soft_model.classes_[all_indices][0]]
        new_Orthopedics_detail_symtoms = new_Orthopedics_detail['토큰키워드'].values
        new_Orthopedics_detail_symtom_vectors = cv2.fit_transform(new_Orthopedics_detail_symtoms)
        new_Orthopedics_detail_symtom_tfidf = tfidf2.fit_transform(new_Orthopedics_detail_symtom_vectors)

        user_vector2 = cv2.transform([' '.join(qustion_key_list)])
        user_tfidf2 = tfidf2.transform(user_vector2)

        # Orthopedics_soft_model - 정형외과 군집 예측에 사용한 모델
        Orthopedics_soft_model.fit(new_Orthopedics_detail_symtom_tfidf, new_Orthopedics_detail['질병명'])
        predicted_probabilities2 = Orthopedics_soft_model.predict_proba(user_tfidf2)[0]
        all_indices2 = np.argsort(predicted_probabilities2)[::-1]

        for i, index in enumerate(all_indices2):
            # print(f'{i+1}. {Orthopedics_soft_model.classes_[index]}')  # {predicted_probabilities2[index]:.4f}
            result.append(Orthopedics_soft_model.classes_[index])

    elif svm.classes_[top_6_dept][0] == '안과':
        new_Eye = new_disease[new_disease['진료과'].str.contains('안과')]

        cv = CountVectorizer()
        tfidf = TfidfTransformer()

        Eye_symptoms = new_Eye['토큰키워드'].values
        Eye_symptom_vectors = cv.fit_transform(Eye_symptoms)
        Eye_symptom_tfidf = tfidf.fit_transform(Eye_symptom_vectors)

        t = Okt()
        q_list = [token for token, pos in t.pos(question, stem=True, norm=True) if pos in ['Noun', 'Adjective']]
        qustion_key_list = [lk for lk in list(set(q_list)) if lk not in stopword]
        user_vector = cv.transform([' '.join(qustion_key_list)])
        user_tfidf = tfidf.transform(user_vector)

        lr = LogisticRegression(multi_class='multinomial', penalty=None, solver='saga', C=10, random_state=0)
        svm = SVC(C=1, gamma='scale', kernel='sigmoid', class_weight='balanced', decision_function_shape='ovo', probability=True, random_state=0)
        nb = MultinomialNB(alpha=0.01, fit_prior=True, force_alpha=True)
        rf = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=2, min_samples_leaf=1, max_features='log2', random_state=0)
        mlp = MLPClassifier(hidden_layer_sizes=(10,10), activation='identity', alpha=0.001, max_iter=100, learning_rate='constant', solver='lbfgs', early_stopping=True, batch_size='auto', warm_start=True, random_state=0)

        Eye_soft_model = VotingClassifier(estimators=[('lr', lr), ('svm', svm), ('nb', nb), ('rf', rf), ('mlp', mlp)], voting="soft")

        # 앙상블 소프트 보팅 모델 학습 - 군집명 예측
        Eye_soft_model.fit(Eye_symptom_tfidf, new_Eye['군집명'])

        # 앙상블 소프트 보팅 모델을 사용하여 군집명 예측
        predicted_probabilities = Eye_soft_model.predict_proba(user_tfidf)[0]

        # 안과 - 예측 군집명
        all_indices = np.argsort(predicted_probabilities)[::-1]
        #print('\n안과 내원 시 예상 군집명')
        #for i, index in enumerate(all_indices):
        #    print(f'{i+1}. {Eye_soft_model.classes_[index]} {predicted_probabilities[index]:.4f}')

        # 군집명 중에서 어떤 질병인지 예측
        # print('\n예상 질병')
        cv2 = CountVectorizer()
        tfidf2 = TfidfTransformer()
        new_Eye_detail = new_Eye.loc[new_Eye['군집명'] == Eye_soft_model.classes_[all_indices][0]]
        new_Eye_detail_symtoms = new_Eye_detail['토큰키워드'].values
        new_Eye_detail_symtom_vectors = cv2.fit_transform(new_Eye_detail_symtoms)
        new_Eye_detail_symtom_tfidf = tfidf2.fit_transform(new_Eye_detail_symtom_vectors)

        user_vector2 = cv2.transform([' '.join(qustion_key_list)])
        user_tfidf2 = tfidf2.transform(user_vector2)

        # Eye_soft_model - 안과 군집 예측에 사용한 모델
        Eye_soft_model.fit(new_Eye_detail_symtom_tfidf, new_Eye_detail['질병명'])
        predicted_probabilities2 = Eye_soft_model.predict_proba(user_tfidf2)[0]
        all_indices2 = np.argsort(predicted_probabilities2)[::-1]

        for i, index in enumerate(all_indices2):
            # print(f'{i+1}. {Eye_soft_model.classes_[index]}')  # {predicted_probabilities2[index]:.4f}
            result.append(Eye_soft_model.classes_[index])

    elif svm.classes_[top_6_dept][0] == '가정의학과':
        new_Family = new_disease[new_disease['진료과'].str.contains('가정의학과')]

        cv = CountVectorizer()
        tfidf = TfidfTransformer()

        Family_symptoms = new_Family['토큰키워드'].values
        Family_symptom_vectors = cv.fit_transform(Family_symptoms)
        Family_symptom_tfidf = tfidf.fit_transform(Family_symptom_vectors)

        t = Okt()
        q_list = [token for token, pos in t.pos(question, stem=True, norm=True) if pos in ['Noun', 'Adjective']]
        qustion_key_list = [lk for lk in list(set(q_list)) if lk not in stopword]
        user_vector = cv.transform([' '.join(qustion_key_list)])
        user_tfidf = tfidf.transform(user_vector)

        Family_nb = MultinomialNB(alpha=0.04, fit_prior=True, force_alpha=True)

        # 인공신경망 모델 학습 - 군집명 예측
        Family_nb.fit(Family_symptom_tfidf, new_Family['군집명'])

        # 인공신경망 모델을 사용하여 군집명 예측
        predicted_probabilities = Family_nb.predict_proba(user_tfidf)[0]

        # 가정의학과 - 예측 군집명
        all_indices = np.argsort(predicted_probabilities)[::-1]
        #print('\n가정의학과 내원 시 예상 군집명')
        #for i, index in enumerate(all_indices):
        #    print(f'{i+1}. {Family_nb.classes_[index]} {predicted_probabilities[index]:.4f}')
        
        # 군집명 중에서 어떤 질병인지 예측
        # print('\n예상 질병')
        cv2 = CountVectorizer()
        tfidf2 = TfidfTransformer()
        new_Family_detail = new_Family.loc[new_Family['군집명'] == Family_nb.classes_[all_indices][0]]
        new_Family_detail_symtoms = new_Family_detail['토큰키워드'].values
        new_Family_detail_symtom_vectors = cv2.fit_transform(new_Family_detail_symtoms)
        new_Family_detail_symtom_tfidf = tfidf2.fit_transform(new_Family_detail_symtom_vectors)

        user_vector2 = cv2.transform([' '.join(qustion_key_list)])
        user_tfidf2 = tfidf2.transform(user_vector2)

        # Family_nb - 가정의학과 군집 예측에 사용한 모델
        Family_nb.fit(new_Family_detail_symtom_tfidf, new_Family_detail['질병명'])
        predicted_probabilities2 = Family_nb.predict_proba(user_tfidf2)[0]
        all_indices2 = np.argsort(predicted_probabilities2)[::-1]

        
        for i, index in enumerate(all_indices2):
            result.append(Family_nb.classes_[index])
            # print(f'{i+1}. {Family_nb.classes_[index]}')  # {predicted_probabilities2[index]:.4f}

    return result