# 2019 빅콘테스트 Analysis분야 챔피언스 리그
**Competition**: [Link](https://www.bigcontest.or.kr/)

## Info
**대회주제**: 리니지 고객(유저) 활동 데이터를 활용하여 잔존 가치를 고려한 이탈 예측 모형 개발

**결과**: 최우수상(NCSOFT상)

**주최**: NIA(한국정보화진흥원)

**주관**:  NCSOFT

**기간**: 2019/07/03 ~ 2019/09/10

## Data
| Train              | Test1              | Test2              | 데이터 내용                                 |
|--------------------|--------------------|--------------------|---------------------------------------------|
| train_label.csv    | -                  | -                  | 대상 유저들의 생존 기간 및 평균 결제 금액   |
| train_activity.csv | test1_activity.csv | test2_activity.csv | 대상 유저의 캐릭터별 활동 이력              |
| train_combat.csv   | test1_combat.csv   | test2_combat.csv   | 대상 유저의 캐릭터별 전투 이력              |
| train_pledge.csv   | test1_pledge.csv   | test2_pledge.csv   | 대상 유저 캐릭터별 소속 혈맹 전투 활동 정보 |
| train_trade.csv    | test1_trade.csv    | test2_trade.csv    | 대상 유저의 캐릭터별 거래 이력              |
| train_payment.csv  | test1_payment.csv  | test2_payment.csv  |          대상 유저의 일별 결제 금액         |

## Source
|   Folder   |                                    Description                                    |
|:----------:|:---------------------------------------------------------------------------------:|
|     raw    |           원본 데이터가 적재될 폴더 (폴더만 생성하고 데이터는 넣지 않음)          |
| preprocess |                   원본 데이터 전처리 코드 및 전처리 결과 데이터                   |
|    model   |                         최종 모델 학습용 코드 및 모델 객체                        |
|   predict  |     테스트 데이터와 모델을 이용하여 최종 답안지를 생성하는 코드 및 최종 답안지    |
|     etc    | 코드 실행 방법에 대한 설명 문서 (readme.txt 혹은 readme.pdf ) 및 서류 심사용 문서 |
