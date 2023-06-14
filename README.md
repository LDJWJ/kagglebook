# kagglebook

책 제목 : 데이터가 뛰어노는 AI 놀이터, 캐글

<figure>
    <img src="https://github.com/LDJWJ/kagglebook/blob/main/bookcover.png" alt="kaggle" width=400 height=400>
</figure>

책 링크 : 
 * https://www.hanbit.co.kr/store/books/look.php?p_code=B4998513859 (한빛 미디어)
 * http://www.yes24.com/Product/Goods/101479127?OzSrank=1 (yes24)
 * https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=270739663 (알라딘)

(독자 의견) frontier1020@naver.com
 * 책과 기타 소중한 의견이 있으시면 메일을 주세요~


### 환경 준비(Window용) 

| 개발환경 | 내용 | 비고 |
|---|---|---|
| Window | [설치하기](https://ldjwj.github.io/kagglebook/pdf_html/1_1_env_simple_2302.html) | 기본 설치 <br> 가상환경 설치없음. 23.02 내용 추가  |
| -- | --  | -- |


* 가상환경은 아나콘다 환경에서 여러 Python버전과 개발 환경을 가질 수 있도록 만든 환경을 말합니다.


### 이 책에서의 라이브러리 버전 정보(실행 확인 2021.03)

| 확인 날짜 | 파이썬 버전내용 | 라이브러리 버전 |
|------|---|---|
| 2021/03 | 파이썬 버전 :  3.8.5 <br> (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]  | Pandas Version - 1.2.2  <br> matplotlib 버전 :  3.3.4 <br> 넘파이(numpy) 버전 :  1.19.5 <br> scikit-learn 버전 :  0.24.1 <br> tensorflow 버전 :  2.4.1 <br> 케라스(keras) 버전 :  2.4.3  <br> xgboost 버전 :  1.3.3 <br> lightgbm 버전 :  3.1.1 <br> hyperopt 버전 :  0.2.5 <br> umap-learn 버전 :  0.5.1  |
| 2023/06 | 파이썬 버전 :  3.9.13 <br>  | Pandas Version - 1.4.4  <br> matplotlib 버전 :  3.5.2 <br> 넘파이(numpy) 버전 :  1.23.5 <br> scikit-learn 버전 :  1.0.2 <br> tensorflow 버전 :  2.12.0 <br> 케라스(keras) 버전 :  2.12.0  <br> xgboost 버전 :  1.7.5 <br> lightgbm 버전 :  3.3.5 <br> hyperopt 버전 :  -- <br> umap-learn 버전 :  -- # umap |

 - 일부 라이브러리 버전이 맞지 않을 경우, 소스코드 일부가 실행이 되지 않을 수 있습니다. 
 - 2023/06 확인 시, anaconda 22년 10월 버전으로 설치.
 - 2023/06 실행 확인을 위해 tensorflow, lightgbm, xgboost은 Anaconda Prompt를 실행 후, 커맨드 환경에서 아래 명령으로 설치 진행.

```
pip install tensorflow
pip install lightgbm
pip install xgboost
```
  

### 이슈 리스트 [상세 내용 확인하기](./error_issue/issue_list.md)
 * 에러 내용
  * 01 keras 설치 후, 불러올때 에러 발생(2021/06/02 추가)
  * 02 bhtsne 모듈 설치가 안되는 이슈(2021.03.13) - 버전 불일치로 판단됨.
  * 03 tensorflow 설치 시, 에러 메시지 발생(2023/06/12 추가)

## 소스 코드 보기
 
 ### ch01 - 타이타닉 대회

 ### ch02
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb) <br> 2023/06 내용 추가|
|------|---|---|
|ch02-01-metrics        |[CODE(.py)](./ch02/ch02-01-metrics.py)|[CODE](./ch02/ch02_01.ipynb) |
|ch02-02-custom-usage   |[CODE(.py)](./ch02/ch02-02-custom-usage.py)|[CODE](./ch02/ch02_02.ipynb)|
|ch02-03-optimize       |[CODE(.py)](./ch02/ch02-03-optimize.py) |[CODE](./ch02/ch02_03.ipynb)|
|ch02-04-optimize-cv    |[CODE(.py)](./ch02/ch02-04-optimize-cv.py)|[CODE](./ch02/ch02_04.ipynb)|
|ch02-05-custom-function|[CODE(.py)](./ch02/ch02-05-custom-function.py)| [CODE](./ch02/ch02_05.ipynb)|

 ### ch03
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch03-01-numerical.py      |[CODE(.py)](./ch03/ch03-01-numerical.py)|[CODE](./ch03/ch03_01_numerical.ipynb) |
|ch03-02-categorical.py    |[CODE(.py)](./ch03/ch03-02-categorical.py)|[CODE](./ch03/ch03_02_categorical.ipynb)|
|ch03-03-multi_tables.py   |[CODE(.py)](./ch03/ch03-03-multi_tables.py) |[CODE](./ch03/ch03_03_multi_tables.ipynb)|
|ch03-04-time_series.py    |[CODE(.py)](./ch03/ch03-04-time_series.py)|[CODE](./ch03/ch03_04_time_series.ipynb)|
|ch03-05-reduction.py      |[CODE(.py)](./ch03/ch03-05-reduction.py)| [CODE](./ch03/ch03_05_reduction.ipynb)|
|ch03-06-reduction-mnist.py|[CODE(.py)](./ch03/ch03-06-reduction-mnist.py)| [CODE](./ch03/ch03_06_reduction_minst.ipynb)|

 ### ch04
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch04-01-introduction.py      |[CODE(.py)](./ch04/ch03-01-numerical.py)|[CODE(노트북)]|
|ch04-02-run_xgb.py   |[CODE(.py)](./ch04/ch04-02-run_xgb.py)|[CODE(노트북)]|
|ch04-03-run_lgb.py   |[CODE(.py)](./ch04/ch04-03-run_lgb.py) |[CODE(노트북)]|
|ch04-04-run_nn.py    |[CODE(.py)](./ch04/ch04-04-run_nn.py)|[CODE(노트북)]|
|ch04-05-run_linear.py      |[CODE(.py)](./ch04/ch04-05-run_linear.py)| [CODE(노트북)]|


 ### ch05
    * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch05-01-validation.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch05/ch05-01-validation.py)|[CODE(노트북)]|
|ch05-02-timeseries.py   |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch05/ch05-02-timeseries.py)|[CODE(노트북)]|

 ### ch06
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch06-01-hopt.py         |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-01-hopt.py)|[CODE(노트북)]|
|ch06-02-hopt_xgb.py     |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-02-hopt_xgb.py)|[CODE(노트북)]|
|ch06-03-hopt_nn.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-03-hopt_nn.py) |[CODE(노트북)]|
|ch06-04-filter.py       |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-04-filter.py)|[CODE(노트북)]|
|ch06-05-embedded.py     |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-05-embedded.py)| [CODE(노트북)]|
|ch06-06-wrapper.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-06-wrapper.py)| [CODE(노트북)]|

 ### ch07
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch07-01-stacking.py         |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch07/ch07-01-stacking.py)|[CODE(노트북)]|
|ch07-02-blending.py     |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch07/ch07-02-blending.py)|[CODE(노트북)]|
|ch07-03-adversarial.py     |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch07/ch07-03-adversarial.py) |[CODE(노트북)]|



## 책에서의 관련 캐글 대회 및 정보 링크

#### ch01
  - 대회의 상위 솔루션 정리 [by sudalairajkumar](https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions)
  - 타이타닉 대회의 솔루션 정리 [by pliptor](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score)

#### ch02
 ### 평가지표로 보는 캐글 대회
   - Instacart Market Basket Analysis [대회로](https://www.kaggle.com/c/allstate-claims-severity)
    - 내용 : 보험 청구는 얼마나 심각한지에 대한 정도의 예측
    - 평가지표 : 평가지표(MAE)
 
  - Instacart Market Basket Analysis [대회로](https://www.kaggle.com/c/human-protein-atlas-image-classification/)
    - 내용 : Instacart 소비자는 어떤 제품을 다시 구매할까?
    - 평가지표 : 평가지표(mean F1 score)
    
  - Santander Product Recommendation [대회로](https://www.kaggle.com/c/santander-product-recommendation)
    - 내용 :Santander Bank  는 개인화 된 제품 추천 
    - 평가지표 : MAP@7 (Mean Average Precision @ 7)

  - Human Protein Atlas Image Classification [대회로](https://www.kaggle.com/c/hpa-single-cell-image-classification)
    - 내용 : 현미경 이미지에서 개별 인간 세포 차이 찾기 (다중 레이블 분류 문제)
    - 평가지표 : Macro F-Score

  - Quora Question Pairs 대회 [대회로](https://www.kaggle.com/c/quora-question-pairs)
    - 내용 : 같은 의도를 가진 질문 쌍을 식별하기
    - 평가지표 : 로그 손실

  - Home Credit Default Risk 대회 [대회로](https://www.kaggle.com/c/home-credit-default-risk)
    - 내용 : 각 신청자가 대출금을 상환 할 수있는 능력을 예측
    - 평가지표 : AUC

  - Two Sigma Connect: Rental Listing Inquiries [대회로](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)
    - 내용 : RentHop의 새 임대 목록에 대한 관심은 얼마나됩니까?
    - 평가지표 : multi-class logloss
   
  - Bosch Production Line Performance [대회로](https://www.kaggle.com/c/bosch-production-line-performance)
    - 내용 : 제조 프로세스 개선을 위한 제조 실패 감소
    - 평가지표 : MCC
    - 
### 시계열 데이터를 다루는 대회
 - Recruit Restaurant Visitor forecasting 대회  [대회로](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting)
   - 내용 : 각 음식점의 일별 손님 수를 알려주고 미래의 손님 수를 예측
 - Santander Product Recommendation 대회 [대회로](https://www.kaggle.com/c/santander-product-recommendation)
   - 내용 : 각 고객의 금융상품 구매 이력을 월 단위로 제공하고 가장 최근 월의 구매상품을 예측
 - Two Sigma financial modeling Challenge 대회 [대회로](https://www.kaggle.com/c/two-sigma-financial-modeling)
   - 내용 : 금융 시장의 특징이 익명화된 시계열 데이터를 주고 지정된 특징의 미래 값을 예측
 - Coupon Purchase Prediction 대회 [대회로](https://www.kaggle.com/c/coupon-purchase-prediction)
   - 내용 : 공동구매형 쿠폰 사이트의 사용자와 과거에 판매된 쿠폰 및 구매 이력 등의 정보를 주고 미래에 각 사용자가 어떤 쿠폰을 구매할지 예측
  
  - 기타 대회
    - 자연어 처리 대회 - Quora Question Pairs 대회
    - 정형 데이터와 광고 이미지 함께 분석 - Avito Demand Prediction Challenge 대회
   
 ### 상위 솔루션
  - Allstate Claims Severity Competition, 2nd Place Winner’s Interview: Alexey Noskov [이동](https://medium.com/kaggle-blog/allstate-claims-severity-competition-2nd-place-winners-interview-alexey-noskov-f4e4ce18fcfc)
  - Home Credit Default Risk - 2nd place solution [이동](https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions?slide=11)

#### ch03
  - [Instacart Market Basket Analysis 2nd place solution](https://www.slideshare.net/kazukionodera7/kaggle-meetup-3-instacart-2nd-placesolution) 
    - 최근 주문 여부를 보여주는 배열을 주고, 가장 최근 기록부터 중요도를 매개 수치로 변환하는 기술 활용
    - 아이템 특별 상품에 주목 - 유기농, 글루텐, 아시아의 아이템에 주목
  - 와이드 포맷 자료 [Link1](https://www.datacamp.com/community/tutorials/long-wide-data-R) [Link2](https://seananderson.ca/2013/10/19/reshape/)
  - Porto Seguro’s Safe Driver Prediction 대회의 1위 솔루션 [Link](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629)
    - 잡음 제거 오토 인코더 사용 
  - 실제로 캐글 Allstate Claims Severity 대회의 2위 솔루션 [Link](https://medium.com/kaggle-blog/allstate-claims-severity-competition-2nd-place-winners-interviewalexey-noskov-f4e4ce18fcfc)
    - 클러스터 중심으로부터 거리를 특징으로 사용
  - 사이킷런의 cluster 모듈 [Link](https://scikit-learn.org/stable/modules/clustering.html)
  - 여러 변수를 조합한 지수 사용 - Recruit Restaurant Visitor Forecasting 대회 20th Solution [Link](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/49328)
  - Avito Demand Prediction Challenge 대회 9위 솔루션 (https://www.slideshare.net/JinZhan/kaggle-avito-demand-prediction-challenge-9th-place-solution-124500050)
     - 중요한 변수인 가격에 대해 상품명, 상품 분류(카테고리), 사용자나 지역 등 다양한 관점에서 평균과의 차와 비율 확인

### 이미지 관련 대회
  * PetFinder.my Adoption Prediction [이동](https://www.kaggle.com/competitions/petfinder-adoption-prediction)
    * 기간 : 2019.03.21 ~ 2019.03.28
    * 평가지표 : RMSE
    * 내용 : PetFinder.my 는 2008년부터 말레이시아 최고의 동물 복지 플랫폼으로 150,000마리 이상의 동물에 대한 데이터베이스를 보유. 애완동물의 입양 가능성을 예측하는 알고리즘을 개발.
    * 애완동물 목록을 기반으로 애완동물의 입양하는 속도를 예측.

2023.06 ~ 
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLDJWJ%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
