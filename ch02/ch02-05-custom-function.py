import numpy as np
import pandas as pd


# -----------------------------------
# 사용자 정의 목적함수의 평가지표 근사에 의한 MAE 최적화
# -----------------------------------

# fair 함수
def fair(preds, dtrain):
    x = preds - dtrain.get_labels()  # 잔차 획득
    c = 1.0                  # fair 함수 파라미터
    den = abs(x) + c         # 그레이디언트 식의 분모 계산
    grad = c * x / den       # 그레이디언트
    hess = c * c / den ** 2  # 이차 미분값
    return grad, hess


# Pseudo-Huber 함수
def psuedo_huber(preds, dtrain):
    d = preds - dtrain.get_labels()  # 잔차 획득
    delta = 1.0                    # Pseudo-Huber 함수 파라미터
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt          # 그레이디언트
    hess = 1 / scale / scale_sqrt  # 이차 미분값
    return grad, hess

# [보강] 추후 이 함수 사용 코드 추가하기
