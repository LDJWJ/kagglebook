import numpy as np
import pandas as pd

# -----------------------------------
# 회귀
# -----------------------------------
# rmse

from sklearn.metrics import mean_squared_error

# y_true - 실젯값, y_pred - 예측값
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.5532

# -----------------------------------
# 이진 분류
# -----------------------------------
# 혼동행렬

from sklearn.metrics import confusion_matrix

# 0, 1로 표현되는 이진 분류의 실젯값과 예측값
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp],
                              [fn, tn]])
print(confusion_matrix1)
# array([[3, 1],
#        [2, 2]])

# 사이킷런의 metrics 모듈의 confusion_matrix로도 작성 가능하지만,
# 혼동행렬의 요소 배치가 다르므로 주의가 필요
confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)
# array([[2, 1],
#        [2, 3]])

# -----------------------------------
# 정확도(accuracy)

from sklearn.metrics import accuracy_score

# 0, 1로 표현되는 이진 분류의 실젯값과 예측값
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625

# -----------------------------------
# logloss

from sklearn.metrics import log_loss

# 0, 1로 나타나는 이진 분류의 실젯값과 예측 확률
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.7136

# -----------------------------------
# 다중 클래스 분류
# -----------------------------------
# multi-class logloss

from sklearn.metrics import log_loss

# 3 클래스 분류의 실젯값과 예측값
y_true = np.array([0, 2, 1, 2, 2])
y_pred = np.array([[0.68, 0.32, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.60, 0.40, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.28, 0.12, 0.60]])
logloss = log_loss(y_true, y_pred)
print(logloss)
# 0.3626

# -----------------------------------
# 다중 레이블 분류
# -----------------------------------
# mean_f1, macro_f1, micro_f1

from sklearn.metrics import f1_score

# 다중 레이블 분류의 실젯값·예측값은 평가지표 계산상으로는 행 데이터 × 클래스의 두
# 값 행렬로 해야 다루기 쉬움
# 실젯값 - [[1,2], [1], [1,2,3], [2,3], [3]]
y_true = np.array([[1, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [0, 1, 1],
                   [0, 0, 1]])

# 예측값 - [[1,3], [2], [1,3], [3], [3]]
y_pred = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

# mean-f1는 행 데이터마다 F1-score를 계산하여 평균을 취함
mean_f1 = np.mean([f1_score(y_true[i, :], y_pred[i, :]) for i in range(len(y_true))])

# macro-f1에서는 행 데이터마다 F1-score를 계산하여 평균을 취함
n_class = 3
macro_f1 = np.mean([f1_score(y_true[:, c], y_pred[:, c]) for c in range(n_class)])

# micro-f1에서는 행 데이터 × 클래스의 쌍으로 TP/TN/FP/FN을 계산하여 F1-score를 구함
micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))

print(mean_f1, macro_f1, micro_f1)
# 0.5933, 0.5524, 0.6250

# scikit-learn 메소드를 사용하여 계산 가능
mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

# -----------------------------------
# 클래스간 순서관계가 있는 다중 클래스 분류
# -----------------------------------
# quadratic weighted kappa

from sklearn.metrics import confusion_matrix, cohen_kappa_score

# quadratic weighted kappa을 계산하는 함수
def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            wij = ((i - j) ** 2.0)
            oij = c_matrix[i, j]
            eij = c_matrix[i, :].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij

    return 1.0 - numer / denom

# y_true는 실젯값 클래스 목록, y_pred는 예측값 클래스 목록
y_true = [1, 2, 3, 4, 3]
y_pred = [2, 2, 4, 4, 5]

# 혼동행렬을 계산
c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

# quadratic weighted kappa를 계산
kappa = quadratic_weighted_kappa(c_matrix)
print(kappa)
# 0.6153

# scikit-learn의 메소드로도 계산 가능
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

# -----------------------------------
# Recommendation(추천)
# -----------------------------------
# MAP@K

# K=3、행의 수는 5개, 클래스는 4종류
K = 3

# 각 행의 실젯값
y_true = [[1, 2], [1, 2], [4], [1, 2, 3, 4], [3, 4]]

# 각 행에 대한 예측값 - K = 3이므로, 일반적으로 각 행에 각각 3개까지 순위를 매겨 예측
y_pred = [[1, 2, 4], [4, 1, 2], [1, 4, 3], [1, 2, 3], [1, 2, 4]]

# 각 행의 average precision을 계산하는 함수
def apk(y_i_true, y_i_pred):
    # y_pred가 K이하의 길이이고 모든 요소가 달라야 함
    assert (len(y_i_pred) <= K)
    assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision

    return sum_precision / min(len(y_i_true), K)

# MAP@K을 계산하는 함수
def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])

# MAP@K을 요청
print(mapk(y_true, y_pred))
# 0.65

# 정답 수가 같아도 순서가 다르면 점수도 다름
print(apk(y_true[0], y_pred[0]))
print(apk(y_true[1], y_pred[1]))
# 1.0, 0.5833
