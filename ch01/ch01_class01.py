import numpy as np
import pandas as pd

# -----------------------------------
# 학습 데이터, 테스트 데이터의 불러오기
# -----------------------------------
# 학습 데이터, 테스트 데이터의 불러오기
train = pd.read_csv('../input/ch01-titanic/train.csv')
test = pd.read_csv('../input/ch01-titanic/test.csv')

# 학습 데이터를 특징과 목적변수로 나누기
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

# 테스트 데이터는 특징만 있으므로, 그대로 사용
test_x = test.copy()

# -----------------------------------
# 특징 추출(피처 엔지니어링)
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 변수 PassengerId를 제거
train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)

# 변수 [Name, Ticket, Cabin]을 제거
train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 범주형 변수에 label encoding 을 적용하여 수치로 변환
for c in ['Sex', 'Embarked']:
    # 학습 데이터를 기반으로 어떻게 변환할지 최적화
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    # 학습 데이터, 테스트 데이터를 변환
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

print(train_x)

print("모델 만들기")
# -----------------------------------
# 모델 만들기
# -----------------------------------
from xgboost import XGBClassifier

# 모델 생성 및 학습 데이터를 이용한 모델 학습
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# 테스트 데이터의 예측 결과를 확률로 출력
pred = model.predict_proba(test_x)[:, 1]
print(pred[0:10])

# 테스트 데이터의 예측 결과를 두개의 값(1,0)으로 변환
pred_label = np.where(pred > 0.5, 1, 0)
print(pred_label[0:10])

# 제출용 파일 작성
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)
# score ：0.76555（여기의 실행 결과가 사용자마다 다를 수 있을 가능성이 있습니다.）

# -----------------------------------
# 모델 검증
# -----------------------------------
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

# 각 fold의 평가 점수를 저장을 위한 빈 리스트 선언
scores_accuracy = []
scores_logloss = []


# 교차 검증(Cross-validation)을 수행
# 01 학습 데이터를 4개로 분할
# 02 그중 하나를 평가용 데이터셋으로 지정
# 03 이후 평가용 데이터의 블록을 하나씩 옆으로 옮겨가며 검증을 수행
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    # 학습 데이터를 학습 데이터와 평가용 데이터셋으로 분할
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 모델 학습을 수행
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    # 평가용 데이터의 예측 결과를 확률로 출력
    va_pred = model.predict_proba(va_x)[:, 1]

    # 평가용 데이터의 점수를 계산
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # 각 fold의 점수를 저장
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

print(scores_logloss)
print(scores_accuracy)

#각 fold의 점수 평균을 출력.
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
# logloss: 0.4270, accuracy: 0.8148（여기의 실행 결과가 사용자마다 다를 수 있을 가능성이 있습니다.）

# -----------------------------------
# 모델 튜닝
# -----------------------------------
import itertools

# 튜닝을 위한 후보 파라미터 값을 준비
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

# 하이퍼파라미터 값의 조합
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

# 각 파라미터의 조합(params)과 그에 대한 점수를 보존(scores)하는 빈 리스트
params = []
scores = []

# 각 파라미터 조합별로 교차 검증(Cross-validation)으로 평가를 수행
for max_depth, min_child_weight in param_combinations:

    score_folds = []
    # 교차 검증(Cross-validation)을 수행
    # 학습 데이터를 4개로 분할한 후,
    # 그중 하나를 평가용 데이터로 삼아 평가. 이를 데이터를 바꾸어 가면서 반복
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        # 학습 데이터를 학습 데이터와 평가용 데이터로 분할
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # 모델의 학습을 수행
        model = XGBClassifier(n_estimators=20, random_state=71,
                              max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # 검증용 데이터의 점수를 계산한 후 저장
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)

    # 각 fold의 점수 평균을 구함
    score_mean = np.mean(score_folds)

    # 파라미터를 조합하고 그에 대한 점수를 저장
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

# 전체 파라미터와 점수 확인
print(params)
print(scores)

# 가장 점수가 좋은 것을 베스트 파라미터로 지정
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# max_depth=7, min_child_weight=2.0의 점수가 가장 좋음.

# -----------------------------------
# 로지스틱 회귀용 특징 작성
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# 원본 데이터를 복사하기
train_x2 = train.drop(['Survived'], axis=1)
test_x2 = test.copy()

# 변수 PassengerId를 제거
train_x2 = train_x2.drop(['PassengerId'], axis=1)
test_x2 = test_x2.drop(['PassengerId'], axis=1)

# 변수 [Name, Ticket, Cabin]을 제거
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 원핫 인코딩(one hot encoding)을 수행
cat_cols = ['Sex', 'Embarked', 'Pclass']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(train_x2[cat_cols].fillna('NA'))

# 원핫 인코딩의 더미 변수의 열이름을 작성
ohe_columns = []
for i, c in enumerate(cat_cols):
    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 원핫 인코딩에 의한 변환을 수행
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# 원핫 인코딩이 수행 후, 원래 변수를 제거
train_x2 = train_x2.drop(cat_cols, axis=1)
test_x2 = test_x2.drop(cat_cols, axis=1)

# 원핫 인코딩을 수행된 변수를 결합
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)
test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)

# 원핫 인코딩된 변수 확인
print(train_x2)

# 수치변수의 결측치를 학습 데이터의 평균으로 채우기
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# 변수Fare를 로그 변환을 수행
train_x2['Fare'] = np.log1p(train_x2['Fare'])
test_x2['Fare'] = np.log1p(test_x2['Fare'])

# -----------------------------------
# 앙상블(ensemble)
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# xgboost 모델
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# 로지스틱 회귀 모델
# xgboost 모델과는 다른 특징을 넣어야 하므로 train_x2, test_x2를 생성
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# 예측 결과의 가중 평균 구하기
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)

# 제출용 파일 작성
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_second_ensemble.csv', index=False)
# score ：0.76794（여기의 실행 결과가 사용자마다 다를 수 있을 가능성이 있습니다.）