import os

import numpy as np
import pandas as pd
import xgboost as xgb

from model import Model
from util import Util


class ModelXGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # 데이터 셋
        validation = va_x is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)

        # 하이퍼파라미터 설정
        params = dict(self.params)
        num_round = params.pop('num_round')

        # 학습
        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist,
                                   early_stopping_rounds=early_stopping_rounds)
        else:
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)

    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limit가 사라지는 것을 막기 위해 pickle로 저장하기로 함.
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)
