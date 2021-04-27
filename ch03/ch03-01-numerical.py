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

# 설명용으로 학습 데이터와 테스트 데이터의 원래 상태를 복제해 두기
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()

# 학습 데이터와 테스트 데이터를 반환하는 함수
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x

# 변환할 수치 변수를 목록에 저장
num_cols = ['age', 'height', 'weight', 'amount',
            'medical_info_a1', 'medical_info_a2', 'medical_info_a3', 'medical_info_b1']

# -----------------------------------
# 표준화
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 학습 데이터를 기반으로 복수 열의 표준화를 정의(평균 0, 표준편차 1)
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# 표준화를 수행한 후 각 열을 치환
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 학습 데이터와 테스트 데이터를 결합한 결과를 기반으로 복수 열의 표준화를 정의
scaler = StandardScaler()
scaler.fit(pd.concat([train_x[num_cols], test_x[num_cols]]))

# 표준화 변환 후 데이터로 각 열을 치환
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import StandardScaler

# 학습 데이터와 테스트 데이터를 각각 표준화(나쁜 예)
scaler_train = StandardScaler()
scaler_train.fit(train_x[num_cols])
train_x[num_cols] = scaler_train.transform(train_x[num_cols])

scaler_test = StandardScaler()
scaler_test.fit(test_x[num_cols])
test_x[num_cols] = scaler_test.transform(test_x[num_cols])

# -----------------------------------
# Min-Max 스케일링
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import MinMaxScaler

# 학습 데이터를 기반으로 여러 열의 최소-최대 스케일링 정의
scaler = MinMaxScaler()
scaler.fit(train_x[num_cols])

# 정규화(0~1) 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 로그 변환
# -----------------------------------
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

# 단순히 값에 로그를 취함
x1 = np.log(x)

# 1을 더한 뒤에 로그를 취함
x2 = np.log1p(x)

# 절댓값의 로그를 취한 후, 원래의 부호를 추가
x3 = np.sign(x) * np.log(np.abs(x))

# -----------------------------------
# Box-Cox 변환
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------

# 양의 정숫값만을 취하는 변수를 변환 대상으로 목록에 저장
# 또한, 결측값을 포함하는 경우는 (~(train_x[c] <= 0.0)).all() 등으로 해야 하므로 주의
pos_cols = [c for c in num_cols if (train_x[c] > 0.0).all() and (test_x[c] > 0.0).all()]

from sklearn.preprocessing import PowerTransformer

# 학습 데이터를 기반으로 복수 열의 박스-칵스 변환 정의
pt = PowerTransformer(method='box-cox')
pt.fit(train_x[pos_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[pos_cols] = pt.transform(train_x[pos_cols])
test_x[pos_cols] = pt.transform(test_x[pos_cols])

# -----------------------------------
# Yeo-Johnson변환
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------

from sklearn.preprocessing import PowerTransformer

# 학습 데이터를 기반으로 복수 열의 여-존슨 변환 정의
pt = PowerTransformer(method='yeo-johnson')
pt.fit(train_x[num_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = pt.transform(train_x[num_cols])
test_x[num_cols] = pt.transform(test_x[num_cols])

# -----------------------------------
# clipping
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
# 열마다 학습 데이터의 1%, 99% 지점을 확인
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

# 1％점 이하의 값은 1%점으로, 99%점 이상의 값은 99%점으로 클리핑
train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)

# -----------------------------------
# binning
# -----------------------------------
x = [1, 7, 5, 4, 6, 3]

# 팬더스 라이브러리의 cut 함수로 구간분할 수행

# bin의 수를 지정할 경우
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0] - 변환된 값은 세 구간(0, 1, 2)를 만들고 원본 x의 값이 어디에 해당되는지 나타냄

# bin의 범위를 지정할 경우(3.0 이하, 3.0보다 크고 5.0보다 이하, 5.0보다 큼)
bin_edges = [-float('inf'), 3.0, 5.0, float('inf')]
binned = pd.cut(x, bin_edges, labels=False)
print(binned)
# [0 2 1 1 2 0] - 변환된 값은 세 구간을 만들고 원본 x의 값이 어디에 해당되는지 나타냄

# -----------------------------------
# 순위로 변환
# -----------------------------------
x = [10, 20, 30, 0, 40, 40]

# 팬더스의 rank 함수로 순위 변환
rank = pd.Series(x).rank()
print(rank.values)
# 시작이 1, 같은 순위가 있을 경우에는 평균 순위가 됨
# [2. 3. 4. 1. 5.5 5.5]

# 넘파이의 argsort 함수를 2회 적용하는 방법으로 순위 변환
order = np.argsort(x)
rank = np.argsort(order)
print(rank)
# 넘파이의 argsort 함수를 2회 적용하는 방법으로 순위 변환
# [1 2 3 0 4 5]

# -----------------------------------
# RankGauss
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import QuantileTransformer

# 학습 데이터를 기반으로 복수 열의 RankGauss를 통한 변환 정의
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

# 변환 후의 데이터로 각 열을 치환
train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])
