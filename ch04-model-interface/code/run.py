import numpy as np
import pandas as pd

from model_nn import ModelNN
from model_xgb import ModelXGB
from runner import Runner
from util import Submission

if __name__ == '__main__':

    params_xgb = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_round': 10000,
        'early_stopping_rounds': 10,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_round'] = 350

    params_nn = {
        'layers': 3,
        # 샘플을 위해 빨리 끝나도록 설정
        'nb_epoch': 5,  # 1000
        'patience': 10,
        'dropout': 0.5,
        'units': 512,
    }

    # 특징 지정
    features = [f'feat_{i}' for i in range(1, 94)]

    # xgboost에 의한 학습 및 예측
    runner = Runner('xgb1', ModelXGB, features, params_xgb)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission('xgb1')

    # 신경망에 의한 학습 예측
    runner = Runner('nn1', ModelNN, features, params_nn)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission('nn1')

    '''
    # (참고）xgboost를 통한 학습 및 예측 - 학습 데이터 전체를 사용하는 경우
    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb_all)
    runner.run_train_all()
    runner.run_test_all()
    Submission.create_submission('xgb1-train-all')
    '''
