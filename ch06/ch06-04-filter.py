# ---------------------------------
# 데이터 등의 사전 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series로 유지합니다.(numpy의 array로 유지하기도 합니다)

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# ---------------------------------
# argsort에 의한 인덱스 정렬
# ---------------------------------
# argsort을 사용하면 배열 값이 작은 순서/큰 순서로 인덱스를 정렬할 수 있음
ary = np.array([10, 20, 30, 0])
idx = ary.argsort()
print(idx)        # 내림차순 - [3 0 1 2]
print(idx[::-1])  # 오름차순 - [2 1 0 3]

print(ary[idx[::-1][:3]])  # 베스트 3을 출력 - [30, 20, 10]

# ---------------------------------
# 상관계수
# ---------------------------------
import scipy.stats as st

# 상관계수
corrs = []
for c in train_x.columns:
    corr = np.corrcoef(train_x[c], train_y)[0, 1]
    corrs.append(corr)
corrs = np.array(corrs)

# 스피어만 상관계수
corrs_sp = []
for c in train_x.columns:
    corr_sp = st.spearmanr(train_x[c], train_y).correlation
    corrs_sp.append(corr_sp)
corrs_sp = np.array(corrs_sp)

# 중요도의 상위를 출력(상위 5개까지).
# np.argsort을 사용하여 값의 순서대로 나열한 인덱스를 취득할 수 있음
idx = np.argsort(np.abs(corrs))[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)

idx2 = np.argsort(np.abs(corrs_sp))[::-1]
top_cols2, top_importances2 = train_x.columns.values[idx][:5], corrs_sp[idx][:5]
print(top_cols2, top_importances2)

# ---------------------------------
# 카이제곱 통계량
# ---------------------------------
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# 카이제곱 통계량
x = MinMaxScaler().fit_transform(train_x)
c2, _ = chi2(x, train_y)

# 중요도의 상윗값을 출력(상위 5개까지).
idx = np.argsort(c2)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)

# ---------------------------------
# 상호정보량
# ---------------------------------
from sklearn.feature_selection import mutual_info_classif

# 상호정보량
mi = mutual_info_classif(train_x, train_y)

# 중요도의 상위를 출력(상위 5개까지).
idx = np.argsort(mi)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)
