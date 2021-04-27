import numpy as np
import pandas as pd

# ---------------------------------
# 랜덤포레스트 특징의 중요도
# ---------------------------------
# train_x는 학습 데이터, train_y는 목적 변수
# 결손값을 취급할 수 없기 때문에 결손값을 보완한 데이터를 불러온다.
train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
# ---------------------------------
from sklearn.ensemble import RandomForestClassifier

# 랜덤 포레스트 모델
clf = RandomForestClassifier(n_estimators=10, random_state=71)
clf.fit(train_x, train_y)
fi = clf.feature_importances_

# 중요도의 상위를 출력
idx = np.argsort(fi)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], fi[idx][:5]
print('random forest importance')
print(top_cols, top_importances)

# ---------------------------------
# xgboost의 특징 중요도
# ---------------------------------
# train_x는 학습 데이터, train_y는 목적 변수
train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
# ---------------------------------
import xgboost as xgb

# xgboost
dtrain = xgb.DMatrix(train_x, label=train_y)
# params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}  # 기존
params = {'objective': 'binary:logistic', 'verbosity': 0, 'random_state': 71}  # 이슈 대응 수정 2021/02/28
num_round = 50
model = xgb.train(params, dtrain, num_round)

# 중요도의 상위를 출력
fscore = model.get_score(importance_type='total_gain')
fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
print('xgboost importance')
print(fscore[:5])


