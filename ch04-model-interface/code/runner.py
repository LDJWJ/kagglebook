import numpy as np
import pandas as pd
from model import Model
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from typing import Callable, List, Optional, Tuple, Union

from util import Logger, Util

logger = Logger()


class Runner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model], features: List[str], params: dict):
        """컨스트럭터

        :param run_name: run의 이름
        :param model_cls: 모델 클래스
        :param features: 특징 목록
        :param params: 하이퍼 파라미터
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.n_fold = 4

    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """교차 검증에서의 fold를 지정하여 학습 및 평가를 실시한다.

        다른 메서드에서 호출하는 것 외에 단체에서도 확인이나 파라미터 조정에 이용하는

        :param i_fold: fold의 번호（모든 경우는 'all'로 한다.）
        :return: （모델의 인스턴스, 레코드의 인덱스, 예측치, 평가에 의한 스코어）
        """
        # 학습 데이터 읽기
        validation = i_fold != 'all'
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        if validation:
            # 학습 데이터・검증 데이터 셋으로 함
            tr_idx, va_idx = self.load_index_fold(i_fold)
            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # 학습을 실행
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # 검증 데이터에 대한 예측 및 평가를 실시
            va_pred = model.predict(va_x)
            score = log_loss(va_y, va_pred, eps=1e-15, normalize=True)

            # 모델, 인덱스, 예측값, 평가를 반환한다.
            return model, va_idx, va_pred, score
        else:
            # 학습 데이터 모두로 학습을 수행
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # 모델을 반환
            return model, None, None, None

    def run_train_cv(self) -> None:
        """ 교차 검증으로 학습・평가를 실시

        학습・평가와 함께 각 fold 모델의 저장, 스코어의 로그 출력에 대해서도 실시
        """
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        # 각 fold로 학습
        for i_fold in range(self.n_fold):
            # 학습을 실시
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # 모델 저장
            model.save_model()

            # 결과를 저장
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 각 fold의 결과를 정리
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 예측 결과 저장
        Util.dump(preds, f'../model/pred/{self.run_name}-train.pkl')

        # 평가 결과 저장
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """교차 검증으로 학습한 각 fold 모델의 평균에 따라 테스트 데이터를 예측

        미리 run_train_cv를 실행해 둘 필요가 있음.
        """
        logger.info(f'{self.run_name} - start prediction cv')

        test_x = self.load_x_test()

        preds = []

        # 각 fold모델에서 예측을 수행
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 예측 평균값을 출력
        pred_avg = np.mean(preds, axis=0)

        # 예측 결과 저장
        Util.dump(pred_avg, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

    def run_train_all(self) -> None:
        """학습 데이터를 전체를 학습, 그 모델을 저장"""
        logger.info(f'{self.run_name} - start training all')

        # 학습 데이터 전체로 학습을 수행
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self) -> None:
        """학습 데이터 전체로 학습한 모델을 통해 테스트 데이터 예측을 실시.

        미리 run_train_all을 실행해 둘 필요가 있음.
        """
        logger.info(f'{self.run_name} - start prediction all')

        test_x = self.load_x_test()

        # 학습 데이터 전체를 통해 학습한 모델로 예측을 수행
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(test_x)

        # 예측 결과 저장
        Util.dump(pred, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """교차 검증으로 fold를 지정하여 모델을 작성을 수행

        :param i_fold: fold의 번호
        :return: 모델의 인스턴스
        """
        # run 이름, fold, 모델 클래스에서 모델 만들기
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params)

    def load_x_train(self) -> pd.DataFrame:
        """학습 데이터의 특징을 가져오기

        :return: 학습 데이터의 특징
        """
        # 학습 데이터를 읽어오기
        # 열명으로 추출하는 이상의 일을 수행하는 경우 이 메서드의 수정 필요
        # 매회 train.csv를 읽는 것은 효율이 나쁘기 때문에, 데이터에 따라 적절히 대응하는 것이 바람직하다(다른 메서드도 마찬가지)
        return pd.read_csv('../input/train.csv')[self.features]

    def load_y_train(self) -> pd.Series:
        """학습 데이터의 목적 변수 가져오기

        :return: 학습 데이터의 목적 변수
        """
        # 목적 변수를 읽어내다
        train_y = pd.read_csv('../input/train.csv')['target']
        train_y = np.array([int(st[-1]) for st in train_y]) - 1
        train_y = pd.Series(train_y)
        return train_y

    def load_x_test(self) -> pd.DataFrame:
        """테스트 데이터의 특징 가져오기

        :return: 테스트 데이터 특징
        """
        return pd.read_csv('../input/test.csv')[self.features]

    def load_index_fold(self, i_fold: int) -> np.array:
        """교차 검증에서의 fold를 지정하여 대응하는 레코드의 인덱스를 반환

        :param i_fold: fold 번호
        :return: fold를 지원하는 레코드의 인덱스
        """
        # 학습 데이터, 검증 데이터를 나누는 인덱스를 반환.
        # 여기에서는 난수를 고정하여 매번 작성하고 있지만 파일에 보존하는 방법도 있음.
        train_y = self.load_y_train()
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=71)
        return list(skf.split(dummy_x, train_y))[i_fold]
