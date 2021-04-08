# ---------------------------------
# 데이터 등 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series의 자료형 사용(numpy의 array로 값을 저장하기도 함.)
train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 학습 데이터를 학습 데이터와 평가용 데이터셋으로 분할
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# xgboost에 있어, 사용자 평가지표와 목적 변수의 예
# （참조）https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss

# 특징과 목적변수를 xgboost의 데이터 구조로 변환
# 학습 데이터의 특징과 목적변수는 tr_x, tr_y
# 검증 데이터의 특징과 목적변수는 va_x, va_y
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)

# 사용자 정의 목적함수(이 경우는 logloss이며, xgboost의 ‘binary:logistic’과 동일)
def logregobj(preds, dtrain):
    labels = dtrain.get_label()           # 실젯값 레이블 획득
    preds = 1.0 / (1.0 + np.exp(-preds))  # 시그모이드 함수
    grad = preds - labels                 # 그래디언트
    hess = preds * (1.0 - preds)          # 시그모이드 함수 미분
    return grad, hess

# 사용자 정의 평가지표(이 경우 오류율)
def evalerror(preds, dtrain):
    labels = dtrain.get_label()           # 실젯값 레이블 획득
    return 'custom-error', float(sum(labels != (preds > 0.0))) / len(labels)

# 하이퍼 파라미터의 설정
# xgboost 버전이 하위버전의 경우, 'verbosity':0을 'silent':1로 변경 후, 실행.
# params = {'silent': 1, 'random_state': 71}
params = {'verbosity': 0, 'random_state': 71}   # xgboost 1.3.3 버전 적용
num_round = 50
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# 모델 학습 실행
bst = xgb.train(params, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)

# 목적함수에 binary:logistic을 지정했을 때와 달리 확률로 변환하기 전 값으로 예측값이 출력되므로 변환이 필요
pred_val = bst.predict(dvalid)
pred = 1.0 / (1.0 + np.exp(-pred_val))
logloss = log_loss(va_y, pred)
print(logloss)

# (참고)일반적인 방법으로 학습하는 경우
# params = {'silent': 1, 'random_state': 71, 'objective': 'binary:logistic'}
params = {'verbosity': 0, 'random_state': 71, 'objective': 'binary:logistic'}   # 현 버전 1.3.0 버전

bst = xgb.train(params, dtrain, num_round, watchlist)

pred = bst.predict(dvalid)
logloss = log_loss(va_y, pred)
print(logloss)
