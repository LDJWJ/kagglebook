import numpy as np
import pandas as pd

# -----------------------------------
# 와이드 포맷, 롱 포맷
# -----------------------------------

# 와이드 포맷의 데이터 읽기
df_wide = pd.read_csv('../input/ch03/time_series_wide.csv', index_col=0)
# 인덱스의 형태를 날짜형으로 변경
df_wide.index = pd.to_datetime(df_wide.index)

print(df_wide.iloc[:5, :3])
'''
              A     B     C
date
2016-07-01  532  3314  1136
2016-07-02  798  2461  1188
2016-07-03  823  3522  1711
2016-07-04  937  5451  1977
2016-07-05  881  4729  1975
'''

# 롱 포맷으로 변환
df_long = df_wide.stack().reset_index(1)
df_long.columns = ['id', 'value']

print(df_long.head(10))
'''
           id  value
date
2016-07-01  A    532
2016-07-01  B   3314
2016-07-01  C   1136
2016-07-02  A    798
2016-07-02  B   2461
2016-07-02  C   1188
2016-07-03  A    823
2016-07-03  B   3522
2016-07-03  C   1711
2016-07-04  A    937
...
'''

# 와이드 포맷으로 되돌림
df_wide = df_long.pivot(index=None, columns='id', values='value')

# -----------------------------------
# lag 특징
# -----------------------------------
# 와이드 포맷의 데이터 세팅
x = df_wide
# -----------------------------------
# x는 와이드 포맷의 데이터 프레임
# 인덱스 = 날짜 등의 시간, 열 = 사용자나 매장 등, 값 = 매출 등 주목 대상 변수를 나타냄

# 1일 전의 값을 획득
x_lag1 = x.shift(1)

# 7일 전의 값을 획득
x_lag7 = x.shift(7)

# -----------------------------------
# shift() 함수로 각각의 날짜 데이터 값을 일정 기간 전의 데이터로 치환(여기서는 1일전)
# 첫 번째 행은 이전 날짜가 없어 NaN(빈 값)이 됨. 두 번째부터는 전날 데이터로 치환
# 변환된 데이터 기준으로 rolling() 함수를 이용. window=3(자신을 포함하여 3개 행)
# 3일 범위의 날짜 기간(자신 포함 이전 3일)의 데이터 평균을 구함. 단, NaN이 하나라도 포함되면 NaN 반환

# 1기전부터 3기간의 이동평균 산출
x_avg3 = x.shift(1).rolling(window=3).mean()

# -----------------------------------
# 모든 날짜를 1일 이전 데이터로 치환한 뒤,
# 변환된 데이터의 지정 시점부터 이전 날짜의 7일간의 범위에서 최댓값을 산출
x_max7 = x.shift(1).rolling(window=7).max()

# -----------------------------------
# 7일 이전, 14일 이전, 21일 이전, 28일 이전의 합의 평균으로 치환
x_e7_avg = (x.shift(7) + x.shift(14) + x.shift(21) + x.shift(28)) / 4.0

# -----------------------------------
# 1일 이후의 값을 취득
x_lead1 = x.shift(-1)

# -----------------------------------
# lag 특징
# -----------------------------------
# 데이터 읽어오기
train_x = pd.read_csv('../input/ch03/time_series_train.csv')
event_history = pd.read_csv('../input/ch03/time_series_events.csv')
train_x['date'] = pd.to_datetime(train_x['date'])
event_history['date'] = pd.to_datetime(event_history['date'])
# -----------------------------------

# train_x는 학습 데이터로 사용자 ID, 날짜를 열로 갖는 데이터 프레임
# event_history는 과거에 개최한 이벤트의 정보로 날짜, 이벤트를 열로 가진 데이터 프레임
# occurrences는 날짜, 세일 개최 여부를 열로 가진 DataFrame이 됨
dates = np.sort(train_x['date'].unique())
occurrences = pd.DataFrame(dates, columns=['date'])
sale_history = event_history[event_history['event'] == 'sale']
occurrences['sale'] = occurrences['date'].isin(sale_history['date'])

# 누적합을 얻어 각 날짜별 누적 출현 횟수를 표시
# occurrences는 날짜, 세일 누적 출현 횟수를 열로 갖는 데이터 프레임이 됨
occurrences['sale'] = occurrences['sale'].cumsum()

# 날짜를 기준으로 학습 데이터와 결합
train_x = train_x.merge(occurrences, on='date', how='left')
