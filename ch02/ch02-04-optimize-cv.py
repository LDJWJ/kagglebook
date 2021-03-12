import numpy as np
import pandas as pd

# -----------------------------------
# out-of-fold에서의 임곗값(threshold)의 최적화
# -----------------------------------
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# 샘플 데이터 생성 준비
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# 실젯값과 예측값을 다음과 같은 train_y, train_pred_prob이었다고 가정
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# 교차 검증 구조로 임곗값을 구함
thresholds = []
scores_tr = []
scores_va = []

kf = KFold(n_splits=4, random_state=71, shuffle=True)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_pred_prob)):
    tr_pred_prob, va_pred_prob = train_pred_prob[tr_idx], train_pred_prob[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 최적화 목적함수를 설정
    def f1_opt(x):
        return -f1_score(tr_y, tr_pred_prob >= x)

    # 학습 데이터로 임곗값을 실시하고 검증 데이터로 평가를 수행
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    threshold = result['x'].item()
    score_tr = f1_score(tr_y, tr_pred_prob >= threshold)
    score_va = f1_score(va_y, va_pred_prob >= threshold)
    print(threshold, score_tr, score_va)

    thresholds.append(threshold)
    scores_tr.append(score_tr)
    scores_va.append(score_va)

# 각 fold의 임곗값 평균을 테스트 데이터에 적용
threshold_test = np.mean(thresholds)
print(threshold_test)
