# ---------------------------------
# 데이터 등 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series로 유지합니다.(numpy의 array로 유지하기도 합니다)

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

import xgboost as xgb


# 코드 동작 확인을 위한 Model 클래스
class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y):
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        self.model = xgb.train(params, dtrain, num_round)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# -----------------------------------
# 모델 학습과 예측
# -----------------------------------
# 모델의 하이퍼파라미터를 지정
params = {'param1': 10, 'param2': 100}

# Model 클래스를 정의
# Model 클래스는 fit로 학습하고 predict로 예측값 확률을 출력

# 모델 정의
model = Model(params)

# 학습 데이터로 모델 학습
model.fit(train_x, train_y)

# 테스트 데이터에 대해 예측 결과를 출력
pred = model.predict(test_x)

# -----------------------------------
# 검증
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 학습 데이터 검증 데이터를 나누는 인덱스를 작성
# 학습 데이터를 4개로 나누고 그중 하나를 검증 데이터로 지정
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 학습 데이터와 검증 데이터로 구분
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 모델 정의
model = Model(params)

# 학습 데이터에 이용하여 모델의 학습 수행
# 모델에 따라서는 검증 데이터를 동시에 제공하여 점수 모니터링
model.fit(tr_x, tr_y)

# 검증 데이터에 대해 예측하고 평가 수행
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# -----------------------------------
# 교차 검증(crossvalidation)
# -----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 학습 데이터를 4개로 나누고 그 중 1개를 검증 데이터로 지정
# 분할한 검증 데이터를 바꾸어가며 학습 및 평가를 4회 실시
scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)

for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = Model(params)
    model.fit(tr_x, tr_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    scores.append(score)

# 교차 검증의 평균 점수를 출력
print(f'logloss: {np.mean(scores):.4f}')
