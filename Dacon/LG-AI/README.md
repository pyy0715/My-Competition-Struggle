## 리더보드
평가 산식 : AUC (사용자로부터 불만 접수가 일어날 확률 예측)
public score : 전체 테스트 데이터 중 33%
private score : 전체 테스트 데이터 중 67%
1차 평가의 최종 순위는 private score로 선정


## 평가 방식
1차 평가 : 리더보드 private ranking
2차 평가 : 리더보드 운영 종료 후 상위 20팀은 제출 양식에 맞춰 코드 및 PPT 제출
- 결과 분석, 비즈니스 분석
- 사용자 불만 접수 원인 분석
- err_data의 err간 관계 해석
- quality_data 수치 해석
- err_data와 quality_data간의 관계 해석 필수 포함

3차 평가 : 제출 받은 코드 및 PPT를 평가하여 10팀 선정 후 온라인 대면 평가
최종 수상 3팀 선정


![ERD](https://t1.daumcdn.net/thumb/R1280x0.fjpg/?fname=http://t1.daumcdn.net/brunch/service/user/aS4g/image/AErKoXhR7E03awcSsW6RVYjARpA.PNG "ERD")

**TODO: 사용자로부터 불만 접수가 일어날 확률 예측**

### Data Description 

- err_data : 사용자가 시스템 작동시 시스템 로그 중에서 **상태와 관련 있는 로그만**을 정제하여 수집(예시, 시스템 연결 상태 및 시스템 강제 리붓 등)

- quality_data : 사용자의 시스템 작동 중 문제가 발생하면 측정 가능한 지표들로 해당 시점으로부터 2시간 단위 수집

---

-> 이러한 문제는 동시간대(분이 같은경우)에서 같은 에러코드가 2번 이상 발생할 경우 quality에서 수집함??

시스템 작동 중 문제가 발생하였을 때, 에러가 상태와 관련이 있지 않은 경우 quality에만 기록이 남음. 
이는 2가지의 인사이트를 주는데 
   1. err_data와 quality_data는 서로 다른 시점을 바라볼 수 있음(quality_data의 시점이 더 빠른 경우)
   2. 상태와 관련이 있지 않은 경우

quality는  문제 발생 시점 이후, 2시간 단위로 데이터를 수집()

문제가 발생할 확률과 문제로부터 불만 접수가 일어날 확률의 차이점

즉 이는 3단계로 구성된다고 볼 수 있는데
사용자 행동 -> 문제가 발생활 확률 -> 발생한 문제로부터 불만을 접수할 확률

먼저 문제가 발생할 확률



train_err와 train_quailty는 로그발생시간이 존재하므로, 시스템에서 수집되는 데이터 

그렇다면 그 로그 발생 시간이 유저 아이디별로 동일하는가?



## Train_ERR
- User당 MODEL_NM을 여러 개를 가질 수 있음.(최대 3개)
- User당 FWVER을 여러 개를 가질 수 있음.(최대 4개)
- 

가정:
문제에 대해서 고객이 느끼는 심각도는 다름
심각도의 정도에 대해서 고객은 불만을 접수할 수도 아닐수도 있음.

학습을 어떻게 시켜야 하는가?


사용자들의 에러 발생 접수 간격은 어떻게 되나연?-> 3days

train, test의 에러 시점은 차이가 있음
test는 50,51주기 있음

errcode가 0,1,connection timeout, B-A8002이 약 88%임

## Train_Quality

Quality별 결측값이 있지만 그에 대한 유저 아이디는 적음 
- Quality_0의 경우, 575개의 아이디
- Quality_2의 경우, 61개의 아이디
- QUailty_5의 경우, 5개의 아이디

activity_user는 3167명
fwver이 결측값이라면, quality_0, quality_2는 무조건 결측값임 
quality_0, quality_2는 펌웨이 버전과 관련이 있는 퀄리티 로그?

model_nm은 fwver들의 집합으로 나타낼 수 있음.


quality의 접수 간격은 10분단위로 정해져있음
err


**크게 나타내자면 모델명 > 펌웨어 버전 > 퀄리티**

quality는 전체적으로 fwver와 관련이 있음.
fwver가 결측값이라면 quality_0, quality_2가 결측값으로 나타나게 됨.
quality_0은 model_4와 관련이 있음 
quality_2은 model_1와 관련이 있음


CV score: 0.81721  / LB Score:0.82021
{'bagging_fraction': 0.6003577815398431,
 'feature_fraction': 0.6029219588618207,
 'max_depth': 9.785412922334,
 'min_child_weight': 1.119822884118094,
 'min_split_gain': 0.5287687991753443,
 'num_leaves': 242.15618532152135,
 'reg_alpha': 2.7332664191931784,
 'reg_lambda': 0.5543840217605306}


lgb_baseline3
 CV_score:0.82070 / LB_Score:0.82416

 1/20 
 version1: 0.82155 / 0.8209


quality가 나온 날짜에 비례해서 err데이터를 학습시킬 수 있나?

0.82365 / 