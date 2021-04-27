# ---------------------------------
# 데이터 등의 사전 준비
# ----------------------------------
import numpy as np
import pandas as pd

# train_x는 학습 데이터, train_y는 목적 변수, test_x는 테스트 데이터
# pandas의 DataFrame, Series로 유지합니다.(numpy의 array로 유지하기도 합니다)
# one-hot encoding된 것을 읽어오기

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# 학습 데이터를 학습 데이터와 검증 데이터로 나눕니다.
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# tensorflow의 경고 억제
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# -----------------------------------
# 신경망의 구현
# -----------------------------------
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# 데이터 스케일링
scaler = StandardScaler()
tr_x = scaler.fit_transform(tr_x)
va_x = scaler.transform(va_x)
test_x = scaler.transform(test_x)

# 신경망 모델 구축
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(train_x.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습의 실행
# 검증 데이터도 모델에 제공하여 학습 진행과 함께 점수가 어떻게 달라지는지 모니터링
batch_size = 128
epochs = 10
history = model.fit(tr_x, tr_y,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(va_x, va_y))

# 검증 데이터의 점수를 확인
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred, eps=1e-7)
print(f'logloss: {score:.4f}')

# 예측
pred = model.predict(test_x)

# -----------------------------------
# early stopping
# -----------------------------------
from keras.callbacks import EarlyStopping

# early stopping의 round를 20으로 설정
# restore_best_weights을 설정하므로 최적의 압축 모델을 사용
epochs = 50
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(tr_x, tr_y,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(va_x, va_y), callbacks=[early_stopping])
pred = model.predict(test_x)
