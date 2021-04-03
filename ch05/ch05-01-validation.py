# ---------------------------------
# 데이터 등의 사전 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series로 유지합니다.(numpy의 array로 유지하기도 합니다)

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

# xgboost에 의한 학습·예측을 하는 클래스
import xgboost as xgb

class Model:
    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        # params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
        params = {'objective': 'binary:logistic', 'verbosity': 0, 'random_state': 71}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# -----------------------------------
# 홀드아웃(hold-out)방법
# -----------------------------------
# 홀드아웃(hold-out)방법으로 검증 데이터의 분할
from sklearn.model_selection import train_test_split

# train_test_split()함수를 이용한 홀드아웃 방법으로 분할
tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y,
                                          test_size=0.25, random_state=71, shuffle=True)

# -----------------------------------
# 홀드아웃(hold-out)방법으로 검증을 수행

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Model 클래스를 정의
# Model 클래스는 fit으로 학습하고 predict로 예측값 확률을 출력

# train_test_split 함수를 이용하여 홀드아웃 방법으로 분할
tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y,
                                          test_size=0.25, random_state=71, shuffle=True)

# 학습 실행, 검증 데이터 예측값 출력, 점수 계산
model = Model()
model.fit(tr_x, tr_y, va_x, va_y)
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(score)

# -----------------------------------
# KFold 클래스를 이용하여 홀드아웃 방법으로 검증 데이터를 분할

from sklearn.model_selection import KFold

# KFold 클래스를 이용하여 홀드아웃 방법으로 분할
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
print(tr_idx, va_idx)
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# 교차 검증
# -----------------------------------
# 교차 검증 방법으로 데이터 분할

from sklearn.model_selection import KFold

# KFold 클래스를 이용하여 교차 검증 분할을 수행
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# 교차 검증을 수행

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Model 클래스를 정의
# Model 클래스는 fit으로 학습하고, predict로 예측값 확률을 출력

scores = []

# KFold 클래스를 이용하여 교차 검증 방법으로 분할
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 학습 실행, 검증 데이터의 예측값 출력, 점수 계산
    model = Model()
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    scores.append(score)

# 각 폴더의 점수 평균을 출력
print(np.mean(scores))

# -----------------------------------
# Stratified K-Fold
# -----------------------------------
from sklearn.model_selection import StratifiedKFold

# StratifiedKFold 클래스를 이용하여 층화추출로 데이터 분할
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x, train_y):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# GroupKFold
# -----------------------------------
# 4건씩 같은 유저가 있는 데이터였다고 가정한다.
train_x['user_id'] = np.arange(0, len(train_x)) // 4
# -----------------------------------

from sklearn.model_selection import KFold, GroupKFold

# user_id열의 고객 ID 단위로 분할
user_id = train_x['user_id']
unique_user_ids = user_id.unique()

# KFold 클래스를 이용하여 고객 ID 단위로 분할
scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_group_idx, va_group_idx in kf.split(unique_user_ids):
    # 고객 ID를 train/valid(학습에 사용하는 데이터, 검증 데이터)로 분할
    tr_groups, va_groups = unique_user_ids[tr_group_idx], unique_user_ids[va_group_idx]

    # 각 샘플의 고객 ID가 train/valid 중 어느 쪽에 속해 있느냐에 따라 분할
    is_tr = user_id.isin(tr_groups)
    is_va = user_id.isin(va_groups)
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]

# (참고)GroupKFold 클래스에서는 셔플과 난수 시드를 지정할 수 없으므로 사용하기 어려움
kf = GroupKFold(n_splits=4)
for tr_idx, va_idx in kf.split(train_x, train_y, user_id):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# leave-one-out
# -----------------------------------
# 데이터가 100건밖에 없는 것으로 간주
train_x = train_x.iloc[:100, :].copy()
# -----------------------------------
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
for tr_idx, va_idx in loo.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
