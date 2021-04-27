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

# 시계열 데이터이며, 시간에 따라 변수 period를 설정했다고 함
train_x['period'] = np.arange(0, len(train_x)) // (len(train_x) // 4)
train_x['period'] = np.clip(train_x['period'], 0, 3)
test_x['period'] = 4

# -----------------------------------
# 시계열 데이터의 홀드아웃(hold-out)방법
# -----------------------------------
# 변수 period를 기준으로 분할하기로 함(0부터 2까지 학습 데이터, 3이 테스트 데이터)
# 변수 period가 1, 2, 3의 데이터를 각각 검증 데이터로 하고 그 이전 데이터를 학습에 사용
is_tr = train_x['period'] < 3
is_va = train_x['period'] == 3
tr_x, va_x = train_x[is_tr], train_x[is_va]
tr_y, va_y = train_y[is_tr], train_y[is_va]

# -----------------------------------
# 시계열 데이터의 교차 검증(시계열에 따라 시행하는 방법)
# -----------------------------------
# 변수 period를 기준으로 분할(0부터 2까지가 학습 데이터, 3이 테스트 데이터)
# 변수 period가 1, 2, 3의 데이터를 각각 검증 데이터로 하고 그 이전 데이터를 학습에 사용

va_period_list = [1, 2, 3]
for va_period in va_period_list:
    is_tr = train_x['period'] < va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]

# (참고)periodSeriesSplit의 경우, 데이터 정렬 순서밖에 사용할 수 없으므로 쓰기 어려움
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=4)
for tr_idx, va_idx in tss.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# -----------------------------------
# 시계열 데이터의 교차 검증(단순하게 시간으로 분할하는 방법)
# -----------------------------------
# 변수 period를 기준으로 분할(0부터 3까지가 학습 데이터, 3이 테스트 데이터).
# 변수 period가 0, 1, 2, 3인 데이터를 각각 검증 데이터로 하고, 그 이외의 학습 데이터를 학습에 사용

va_period_list = [0, 1, 2, 3]
for va_period in va_period_list:
    is_tr = train_x['period'] != va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]
