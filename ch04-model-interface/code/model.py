import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional


class Model(metaclass=ABCMeta):

    def __init__(self, run_fold_name: str, params: dict) -> None:
        """ Constructor

        :param run_fold_name: run의 이름과 fold의 번호를 조합한 이름
        :param params: 하이퍼 파라미터
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series,
              va_x: Optional[pd.DataFrame] = None,
              va_y: Optional[pd.Series] = None) -> None:
        """모델을 학습하고, 학습이 끝난 모델을 저장한다.

        :param tr_x: 학습 데이터의 특징
        :param tr_y: 학습 데이터의 목적 변수
        :param va_x: 밸리데이션 데이터의 특징
        :param va_y: 밸리데이션 데이터의 목적 변수
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        """학습이 끝난 모델에서의 예측치를 반환한다.

        :param te_x: 밸리데이션 데이터나 테스트 데이터의 특징
        :return: 예측치
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """모델의 저장을 실시한다"""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """모델을 불러오기"""
        pass
