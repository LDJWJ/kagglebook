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

# neural net용의 데이터
train_nn = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x_nn = train_nn.drop(['target'], axis=1)
train_y_nn = train_nn['target']
test_x_nn = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# ---------------------------------
# 스태킹(stacking)
# ----------------------------------
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# models.py에 Model1Xgb, Model1NN, Model2Linear을 정의하는 것으로 함
# 각 클래스는 fit로 학습하고 predict로 예측값 확률을 출력

from models import Model1Xgb, Model1NN, Model2Linear

# 학습 데이터에 대한 ‘목적변수를 모르는’예측값과, 테스트 데이터에 대한 예측값을 반환하는 함수
def predict_cv(model, train_x, train_y, test_x):
    preds = []
    preds_test = []
    va_idxes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=71)

    # 교차 검증으로 학습・예측하여 예측값과 인덱스를 보존
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # 검증에 대한 예측값을 연결하고 이후 원래 순서로 정렬
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # 테스트 데이터에 대한 예측값의 평균 획득
    preds_test = np.mean(preds_test, axis=0)
    return pred_train, preds_test


# 1계층 모델
# pred_train_1a, pred_train_1b는 학습 데이터의 검증에서의 예측값
# pred_test_1a, pred_test_1b는 테스트 데이터의 예측값
model_1a = Model1Xgb()
pred_train_1a, pred_test_1a = predict_cv(model_1a, train_x, train_y, test_x)

model_1b = Model1NN()
pred_train_1b, pred_test_1b = predict_cv(model_1b, train_x_nn, train_y, test_x_nn)

# 1계층 모델의 평가
print(f'logloss: {log_loss(train_y, pred_train_1a, eps=1e-7):.4f}')
print(f'logloss: {log_loss(train_y, pred_train_1b, eps=1e-7):.4f}')

# 예측값을 특징으로 데이터 프레임을 작성
train_x_2 = pd.DataFrame({'pred_1a': pred_train_1a, 'pred_1b': pred_train_1b})
test_x_2 = pd.DataFrame({'pred_1a': pred_test_1a, 'pred_1b': pred_test_1b})

# 2계층 모델
# pred_train_2는 2계층 모델의 학습 데이터로 교차 검증에서의 예측값
# pred_test_2는 2계층 모델의 테스트 데이터 예측값
model_2 = Model2Linear()
pred_train_2, pred_test_2 = predict_cv(model_2, train_x_2, train_y, test_x_2)
print(f'logloss: {log_loss(train_y, pred_train_2, eps=1e-7):.4f}')
