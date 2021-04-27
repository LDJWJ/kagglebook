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

# 학습 데이터를 학습 데이터와 밸리데이션 데이터로 나눕니다.
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# xgboost의 구현
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss

# 특징과 목적변수를 xgboost의 데이터 구조로 변환
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)

# 하이퍼파라미터 설정
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
num_round = 50

# 학습의 실행
# 검증 데이터도 모델에 제공하여 학습 진행과 함께 점수가 어떻게 달라지는지 모니터링
# watchlist로 학습 데이터 및 검증 데이터를 준비
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist)

# 검증 데이터의 점수를 확인
va_pred = model.predict(dvalid)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# 예측 - 두 값(0 or 1)의 예측이 아닌 양성(1)일 확률을 출력
pred = model.predict(dtest)

# -----------------------------------
# 학습 데이터와 검증 데이터의 점수를 모니터링
# -----------------------------------
# 모니터링을 logloss로 수행. early_stopping_rounds를 20라운드로 설정
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71,
          'eval_metric': 'logloss'}
num_round = 500
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round,
                  evals=watchlist,
                  early_stopping_rounds=20)

# 최적의 결정 트리의 개수로 예측
pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
