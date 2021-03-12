# ---------------------------------
# 데이터 등 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series의 자료형 사용(numpy의 array로 값을 저장하기도 함.)

train = pd.read_csv('../input/sample-data/train.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test.csv')


# 설명용으로 학습 데이터와 테스트 데이터의 원래 상태를 복제해 두기
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()


# 학습 데이터와 테스트 데이터를 반환하는 함수
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x

# 변환할 수치 변수를 목록에 저장
cat_cols = ['sex', 'product', 'medical_info_b2', 'medical_info_b3']

# -----------------------------------
# one-hot encoding
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------

# 학습 데이터와 테스트 데이터를 결합하여 get_dummies를 통한 one-hot encoding을 수행
all_x = pd.concat([train_x, test_x])
all_x = pd.get_dummies(all_x, columns=cat_cols)

# 학습 데이터와 테스트 데이터의 재분할
train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)

# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoder로 인코딩
ohe = OneHotEncoder(sparse=False, categories='auto')
ohe.fit(train_x[cat_cols])

# 가변수의 컬럼명 생성
columns = []
for i, c in enumerate(cat_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 생성된 가변수를 데이터 프레임으로 변환
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[cat_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[cat_cols]), columns=columns)

# 나머지 변수와의 결합
train_x = pd.concat([train_x.drop(cat_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(cat_cols, axis=1), dummy_vals_test], axis=1)

# -----------------------------------
# label encoding
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 범주형 변수를 for문 루프하여 반복적으로 레이블 인코딩 수행
for c in cat_cols:
    # 학습 데이터에 근거하여 정의한 후에 데이터 변환
    le = LabelEncoder()
    le.fit(train_x[c])
    train_x[c] = le.transform(train_x[c])
    test_x[c] = le.transform(test_x[c])

# -----------------------------------
# feature hashing
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.feature_extraction import FeatureHasher

# 범주형 변수를 반복적으로 특징 해싱 처리
for c in cat_cols:

    # FeatureHasher의 사용법은 다른 encoder와 조금 달라짐
    fh = FeatureHasher(n_features=5, input_type='string')

    # 변수를 문자열로 변환한 후 FeatureHasher 적용
    hash_train = fh.transform(train_x[[c]].astype(str).values)
    hash_test = fh.transform(test_x[[c]].astype(str).values)

    # 데이터 프레임으로 변환
    hash_train = pd.DataFrame(hash_train.todense(), columns=[f'{c}_{i}' for i in range(5)])
    hash_test = pd.DataFrame(hash_test.todense(), columns=[f'{c}_{i}' for i in range(5)])

    # 원래의 데이터 프레임과 결합
    train_x = pd.concat([train_x, hash_train], axis=1)
    test_x = pd.concat([test_x, hash_test], axis=1)

# 원래의 범주형 변수 삭제
train_x.drop(cat_cols, axis=1, inplace=True)
test_x.drop(cat_cols, axis=1, inplace=True)

# -----------------------------------
# frequency encoding
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
# for문을 이용한 변수를 반복하여 프리퀀시 인코딩 수행
for c in cat_cols:
    freq = train_x[c].value_counts()
    # 카테고리 출현 횟수로 치환
    train_x[c] = train_x[c].map(freq)
    test_x[c] = test_x[c].map(freq)

# -----------------------------------
# target encoding
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# for문을 이용한 변수를 반복하여 타깃 인코딩 수행
for c in cat_cols:
    # 학습 데이터 전체에서 각 범주별 타깃 평균을 계산
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    target_mean = data_tmp.groupby(c)['target'].mean()
    # 테스트 데이터의 카테고리 변경
    test_x[c] = test_x[c].map(target_mean)

    # 학습 데이터 변환 후 값을 저장하는 배열을 준비
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 학습 데이터 분할
    kf = KFold(n_splits=4, shuffle=True, random_state=72)
    for idx_1, idx_2 in kf.split(train_x):
        # 아웃 오브 폴드로 각 범주형 목적변수 평균 계산
        target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        # 변환 후의 값을 날짜 배열에 저장
        tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

    # 변환 후의 값을 날짜 배열에 저장
    train_x[c] = tmp

# -----------------------------------
# target encoding - 교차 검증의 각 fold의 경우
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# 교차 검증 폴드마다 타깃 인코딩 다시 적용
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):

    # 학습 데이터에서 학습 데이터와 검증 데이터 구분
    tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 변수를 반복하여 타깃 인코딩 수행
    for c in cat_cols:
        # 학습 데이터 전체에서 각 범주별 타깃 평균을 계산
        data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # 검증 데이터의 카테고리 치환
        va_x.loc[:, c] = va_x[c].map(target_mean)

        # 학습 데이터 변환 후 값을 저장하는 배열 준비
        tmp = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72)
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            # 아웃 오브 폴드에서 각 범주별 목적변수 평균 계산
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 변환 후 값을 날짜 배열에 저장
            tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)

        tr_x.loc[:, c] = tmp

    # 필요에 따라 encode된 특징을 저장하고 나중에 읽을 수 있도록 해둠.

# -----------------------------------
# target encoding - 교차 검증의 fold와 target encoding의 fold 분할을 맞추는 경우
# -----------------------------------
# 데이터 읽어오기
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# 교차 검증의 폴드를 정의
kf = KFold(n_splits=4, shuffle=True, random_state=71)

# 변수를 루프하여 타깃 인코딩 수행
for c in cat_cols:

    # 타깃을 추가
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    # 변환 후 값을 저장하는 배열을 준비
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 학습 데이터에서 검증 데이터를 나누기
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        # 학습 데이터에 대해 각 범주별 목적변수 평균 계산
        target_mean = data_tmp.iloc[tr_idx].groupby(c)['target'].mean()
        # 검증 데이터에 대해 변환 후 값을 날짜 배열에 저장
        tmp[va_idx] = train_x[c].iloc[va_idx].map(target_mean)

    # 변환 후의 데이터로 원래의 변수를 변경
    train_x[c] = tmp
