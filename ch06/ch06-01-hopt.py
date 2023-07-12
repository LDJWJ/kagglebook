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

# 학습 데이터를 학습 데이터와 검증 데이터로 나눔
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# xgboost로 학습 및 예측을 수행
import xgboost as xgb

class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        #params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
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
# 탐색할 파라미터의 공간의 지정
# -----------------------------------
# hp.choice에서는 복수의 선택사항에서 고르기
# hp.uniform에서는 하한상한을 지정한 동일분포로부터 추출. 인수는 하한상한
# hp.quniform에서는 하한상한을 지정한 균등분포 중 일정 간격마다의 점으로부터 추출. 인수는 하한상한간격
# hp.loguniform에서는 하한상한을 지정한 로그을 취한 값이 균등분포를 따르는 분포로부터 추출. 인수는 하한상한의 로그를 취한 값

# 만약 moduleNotFoundError 발생시 hyperopt 설치 후 진행
# 2021/04/27 hyperopt 설치 확인 : 0.2.5
# 2023/06/20 hyperopt 설치 확인 : 0.2.7
# pip install hyperopt
from hyperopt import hp

space = {
    'activation': hp.choice('activation', ['prelu', 'relu']),
    'dropout': hp.uniform('dropout', 0, 0.2),
    'units': hp.quniform('units', 32, 256, 32),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.01)),
}

# -----------------------------------
# hyperopt을 사용한 파라미터 탐색
# -----------------------------------
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import log_loss


def score(params):
    # 매개변수를 주었을 때, 최소화하는 평가지표를 지정
    # 구체적으로는 모델에 매개변수를 지정하여 학습예측하는 경우의 점수를 반환

    # max_depth의 형을 정수형으로 수정
    params['max_depth'] = int(params['max_depth'])

    # Model 클래스를 정의하는 것으로 함
    # Model 클래스는 fit으로 학습하고 predict로 예측값의 확률을 출력
    model = Model(params)
    model.fit(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    print(f'params: {params}, logloss: {score:.4f}')

    # 정보를 기록
    history.append((params, score))

    return {'loss': score, 'status': STATUS_OK}

# 탐색할 매개변수의 공간을 지정
space = {
    'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
    'max_depth': hp.quniform('max_depth', 3, 9, 1),
    'gamma': hp.quniform('gamma', 0, 0.4, 0.1),
}

# hyperopt에 의한 매개변수 탐색 실행
max_evals = 10
trials = Trials()
history = []
fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

# 기록한 정보에서 매개변수와 점수를 출력
# (trials에서도 정보를 취득할 수 있지만 매개변수의 취득이 다소 어려움)
history = sorted(history, key=lambda tpl: tpl[1])
best = history[0]
print(f'best params:{best[0]}, score:{best[1]:.4f}')
