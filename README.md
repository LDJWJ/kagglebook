# kagglebook
 
### 이 책에서의 라이브러리 버전 정보(실행 확인 2021.03)
 - 파이썬 버전 :  3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
 - 팬더스 버전 :  1.2.2
 - matplotlib 버전 :  3.3.4
 - 넘파이 버전 :  1.19.5
 - scikit-learn 버전 :  0.24.1
 - tensorflow 버전 :  2.4.1
 - 케라스 버전 :  2.4.3
 - xgboost 버전 :  1.3.3
 - lightgbm 버전 :  3.1.1
 - hyperopt 버전 :  0.2.5
 - 일부 라이브러리 버전이 맞지 않을 경우, 소스코드 일부가 실행이 되지 않을 수 있습니다. 

### 환경 준비(Window용)
 - 01 아나콘다 설치 - [PDF](https://ldjwj.github.io/kagglebook/pdf_html/01_anaconda_install.pdf) [HTML](https://ldjwj.github.io/kagglebook/pdf_html/1_1_anaconda_install_202103.html)
   - 최신버전 다운로드 및 파일명
     - https://www.anaconda.com/products/individual
     - 파일명 : Anaconda3-2020.11-Windows-x86_64.exe
     - 파일 설치 시 확인 : conda 버전 4.8.2, python 3.7.6
     - (명령어) python --version, conda --version
   - 이전버전 사용시
     - 다운로드 URL : https://repo.anaconda.com/archive/
     - 파일명 : Anaconda3-2020.02-Windows-x86_64
     - 파일 설치 시 확인 : conda 버전 4.8.2, python 3.7.6
     - (명령어) python --version, conda --version
 - 02 가상 환경 설치(가상환경을 사용하지 않을 시, 2번은 건너뛰어도 됨.)
   - anaconda prompt 실행
   - 가상환경 설치
     - (명령어) conda create  --name  tf2x python=3.8.5
   - 가상환경 활성화
     - (명령어) conda activate tf2x   # 가상 환경 활성화
   - tensorflow, keras 및 기본 라이브러리 설치
     - (명령어) pip  install  tensorflow
     - (명령어) pip install keras seaborn pandas jupyter matplotlib scikit-learn
     - (명령어) pip install xgboost
     - (명령어) pip install lightgbm
 - 03 Pycharm 설치 및 환경 설정
   - URL : https://www.jetbrains.com/pycharm/download/#section=windows
 - 04 github 소스 코드 다운로드
   - https://github.com/LDJWJ/kagglebook 로 이동
   - CODE 버튼 선택 - Download Zip 선택 후, 소스 코드 다운로드
   - 내가 원하는 위치로 zip 파일을 이동 후, 압축을 파일을 풀어준다.
 - 05 다운로드 된 소스코드를 pycharm에서 폴더를 선택 후, 불러오기
   - 메뉴 [File]-[Open...]선택 후, 소스코드 폴더를 선택 후, OK선택
   - 각각의 상황에 따라 폴더를 열어준다. 여기서는 Attach로 여러 폴더를 함께 사용.
   - 파일을 하나 선택해서 진행해 본다.

 - [설치 영상 추가 예정] [아나콘다 설치](Link1) [가상환경설치](Link2) [Pycharm 설치 및 소스코드 불러오기](Link3)

### 환경준비(MAC용)
 - 추가 예정


### 환경준비(Linux용)
 - 추가 예정


### 소스 코드 실행 이슈
 - bhtsne 모듈 설치가 안되는 이슈(2021.03.13) - 버전 불일치로 판단됨.


## 소스 코드(Kaggle Notebook(클라우드) 환경 실행 코드)
 - [캐글의 노트북 환경에서 시작하기] [유튜브-추가계획]() [네이버tv-추가계획]() 
 - ch01 - 타이타닉 대회
  - 01
  - 02
  - 03
  -
  
 - ch02 - *** 대회
 
 - ch03 - *** 대회

 - ch04 - *** 대회

 - ch05 - *** 대회

 - ch06 - *** 대회

 - ch07 - *** 대회
 



## 책에서의 관련 링크

#### ch01
  - 대회의 상위 솔루션 정리 [by sudalairajkumar](https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions)
  - 타이타닉 대회의 솔루션 정리 [by pliptor](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score)

#### ch02

#### ch03

#### ch04

#### ch05

#### ch06

#### ch07
