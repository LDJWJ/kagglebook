# ---------------------------------
# 데이터 등 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series의 자료형 사용(numpy의 array로 값을 저장하기도 함.)

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# 설명용으로 학습 데이터와 테스트 데이터의 원래 상태를 복제해 두기
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 표준화한 학습 데이터와 테스트 데이터를 반환하는 함수
def load_standarized_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    return pd.DataFrame(train_x), pd.DataFrame(test_x)


# MinMax 스케일링을 수행한 학습 데이터와 테스트 데이터를 반환하는 함수
def load_minmax_scaled_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()

    # Min-Max Scaling 진행
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([train_x, test_x], axis=0))
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    return pd.DataFrame(train_x), pd.DataFrame(test_x)


# -----------------------------------
# PCA
# -----------------------------------
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
# -----------------------------------
# PCA
from sklearn.decomposition import PCA

# 데이터는 표준화 등의 스케일을 갖추기 위한 전처리가 이루어져야 함

# 학습 데이터를 기반으로 PCA에 의한 변환을 정의
pca = PCA(n_components=5)
pca.fit(train_x)

# 변환 적용
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)

# -----------------------------------
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
# -----------------------------------
# TruncatedSVD
from sklearn.decomposition import TruncatedSVD

# 데이터는 표준화 등의 스케일을 갖추기 위한 전처리가 이루어져야 함

# 학습 데이터를 기반으로 SVD를 통한 변환 정의
svd = TruncatedSVD(n_components=5, random_state=71)
svd.fit(train_x)

# 변환 적용
train_x = svd.transform(train_x)
test_x = svd.transform(test_x)

# -----------------------------------
# NMF
# -----------------------------------
# 비음수의 값이기 때문에 MinMax스케일링을 수행한 데이터를 이용
train_x, test_x = load_minmax_scaled_data()
# -----------------------------------
from sklearn.decomposition import NMF

# 데이터는 음수가 아닌 값으로 구성

# 학습 데이터를 기반으로 NMF에 의한 변환 정의
model = NMF(n_components=5, init='random', random_state=71)
model.fit(train_x)

# 변환 적용
train_x = model.transform(train_x)
test_x = model.transform(test_x)

# -----------------------------------
# LatentDirichletAllocation
# -----------------------------------
# MinMax스케일링을 수행한 데이터를 이용
# 카운트 행렬은 아니지만, 음수가 아닌 값이면 계산 가능
train_x, test_x = load_minmax_scaled_data()
# -----------------------------------
from sklearn.decomposition import LatentDirichletAllocation

# 데이터는 단어-문서의 카운트 행렬 등으로 함

# 학습 데이터를 기반으로 LDA에 의한 변환을 정의
model = LatentDirichletAllocation(n_components=5, random_state=71)
model.fit(train_x)

# 변환 적용
train_x = model.transform(train_x)
test_x = model.transform(test_x)

# -----------------------------------
# LinearDiscriminantAnalysis
# -----------------------------------
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
# -----------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 데이터는 단어-문서의 카운트 행렬 등으로 함

# 학습 데이터를 기반으로 LDA에 의한 변환을 정의
lda = LDA(n_components=1)
lda.fit(train_x, train_y)

# 변환 적용
train_x = lda.transform(train_x)
test_x = lda.transform(test_x)

# -----------------------------------
# t-sne
# -----------------------------------
# 현 버전에 설치 이슈가 있어, 확인 중.
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
# -----------------------------------
import bhtsne

# 데이터는 표준화 등의 스케일을 갖추기 위한 전처리가 이루어져야 함

# t-sne에 의한 변환
data = pd.concat([train_x, test_x])
embedded = bhtsne.tsne(data.astype(np.float64), dimensions=2, rand_seed=71)

# -----------------------------------
# UMAP
# -----------------------------------
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
# -----------------------------------
import umap

# 데이터는 표준화 등의 스케일을 갖추는 전처리가 이루어져야 함

# 학습 데이터를 기반으로 UMAP에 의한 변환을 정의
um = umap.UMAP()
um.fit(train_x)

# 변환 적용
train_x = um.transform(train_x)
test_x = um.transform(test_x)

# -----------------------------------
# クラスタリング
# -----------------------------------
# 표준화된 데이터를 사용
train_x, test_x = load_standarized_data()
# -----------------------------------
from sklearn.cluster import MiniBatchKMeans

# 데이터는 표준화 등의 스케일을 갖추는 전처리가 이루어져야 함

# 학습 데이터를 기반으로 Mini-Batch K-Means를 통한 변환 정의
kmeans = MiniBatchKMeans(n_clusters=10, random_state=71)
kmeans.fit(train_x)

# 해당 클러스터를 예측
train_clusters = kmeans.predict(train_x)
test_clusters = kmeans.predict(test_x)

# 각 클러스터 중심까지의 거리를 출력
train_distances = kmeans.transform(train_x)
test_distances = kmeans.transform(test_x)
