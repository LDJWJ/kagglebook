import numpy as np
import pandas as pd

# -----------------------------------
# 임곗값(threshold)의 최적화
# -----------------------------------
from sklearn.metrics import f1_score
from scipy.optimize import minimize

# 행 데이터 데이터 생성 준비
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# 실젯값과 예측값을 다음과 같은 train_y, train_pred_prob이었다고 가정
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# 임곗값(threshold)을 0.5로 하면, F1은 0.722
init_threshold = 0.5
init_score = f1_score(train_y, train_pred_prob >= init_threshold)
print(init_threshold, init_score)

# 최적화의 목적함수를 설정
def f1_opt(x):
    return -f1_score(train_y, train_pred_prob >= x)


# scipy.optimize의 minimize 메소드에서 최적의 임곗값 구하기
# 구한 최적의 임곗값을 바탕으로 F1을 구하면 0.756이 됨
result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
best_threshold = result['x'].item()
best_score = f1_score(train_y, train_pred_prob >= best_threshold)
print(best_threshold, best_score)
