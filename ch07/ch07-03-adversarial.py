# ---------------------------------
# 데이터 등의 사전 준비
# ----------------------------------
import numpy as np
import pandas as pd

# 데이터의 작성(랜덤 데이터로 하고 있음.)
rand = np.random.RandomState(71)
train_x = pd.DataFrame(rand.uniform(0.0, 1.0, (10000, 2)), columns=['model1', 'model2'])
adv_train = pd.Series(rand.uniform(0.0, 1.0, 10000))
w = np.array([0.3, 0.7]).reshape(1, -1)
train_y = pd.Series((train_x.values * w).sum(axis=1) > 0.5)

# ---------------------------------
# adversarial stochastic blending
# ----------------------------------
# 모델의 예측값을 가중평균하는 가중치 값을 적대적 검증(adversarial validation)으로 구함
# train_x: 각 모델에 의한 확률 예측값(실제로는 순위로 변환한 것을 사용)
# train_y: 목적변수
# adv_train: 학습 데이터의 테스트 데이터다움을 확률로 나타낸 값

from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

n_sampling = 50      # 샘플링 횟수
frac_sampling = 0.5  # 샘플링에서 학습 데이터를 추출하는 비율


def score(x, data_x, data_y):
    # 평가지표는 AUC로 함
    y_prob = data_x['model1'] * x + data_x['model2'] * (1 - x)
    return -roc_auc_score(data_y, y_prob)

# 샘플링으로 가중평균의 가중치 값을 구하는 작업을 반복
results = []
for i in range(n_sampling):
    # 샘플링을 수행
    seed = i
    idx = pd.Series(np.arange(len(train_y))).sample(frac=frac_sampling, replace=False,
                                                    random_state=seed, weights=adv_train)
    x_sample = train_x.iloc[idx]
    y_sample = train_y.iloc[idx]

    # 샘플링한 데이터에 대하여 가중평균의 가중치 값을 최적화로 구하기
    # 제약식을 갖도록 알고리즘은 COBYLA를 선택
    init_x = np.array(0.5)
    constraints = (
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda x: 1.0 - x},
    )
    result = minimize(score, x0=init_x,
                      args=(x_sample, y_sample),
                      constraints=constraints,
                      method='COBYLA')
    results.append((result.x, 1.0 - result.x))

# model1, model2의 가중평균의 가중치
results = np.array(results)
w_model1, w_model2 = results.mean(axis=0)
