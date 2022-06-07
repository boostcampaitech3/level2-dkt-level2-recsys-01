# Deep Knowledge Tracing (DKT)


## 👋 팀원 소개

|                                                  [김동현](https://github.com/donghyyun)                                                   |                                                                          [임지원](https://github.com/sophi1127)                                                                           |                                                 [이수연](https://github.com/coding-groot)                                                  |                                                                        [진상우](https://github.com/Jin-s-work)                                                                         |                                                                         [심재정](https://github.com/Jaejeong98)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![그림1](https://user-images.githubusercontent.com/61958748/172278471-584ffaf5-ea6d-4e63-ae77-7cac4dbae899.png)| ![그림2](https://user-images.githubusercontent.com/61958748/172278474-f2d54e27-898b-4142-af78-b0e370e43ffc.png)| <img width="140" alt="그림3" src="https://user-images.githubusercontent.com/61958748/172278478-f3bbd8ce-3616-4c37-8fa6-4247e20b469e.png">| ![그림4](https://user-images.githubusercontent.com/61958748/172278482-a591c2e4-f4b7-4edf-a390-9e875c2c4226.png)| ![그림5](https://user-images.githubusercontent.com/61958748/172278489-00773bd6-080f-41ec-b828-24f4dabc5f98.png)|


## Contribution

+ [`김동현`](https://github.com/donghyyun) &nbsp; EDA • Feature Engineering • CatBoost

+ [`임지원`](https://github.com/sophi1127) &nbsp; EDA • Feature Engineering • LightGBM • SAKT • wandb

+ [`이수연`](https://github.com/coding-groot) &nbsp; EDA • LightGCN • Ensemble

+ [`진상우`](https://github.com/Jin-s-work) &nbsp; LightGCN • wandb

+ [`심재정`](https://github.com/Jaejeong98) &nbsp; LSTM • LSTM Attention • Last query • wandb


## DKT란?
DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다.

<p align="center">
  <img src="image/지식상태.png" width="800">
</p>

DKT를 통해 단순히 시험 성적을 알려주는 것을 넘어서 해당 분야에 대해 얼마나 이해하고 있는지 측정 가능합니다. 또한 이러한 이해도를 활용하여 아직 풀지 않은 문제의 정답 여부를 예측할 수 있습니다.

<p align="center">
  <img src="image/시나리오.png" width="500">
</p>

해당 프로젝트는 Iscream 데이터셋을 활용하여 학생이 마지막으로 주어진 문제의 정답을 맞출 수 있는지 여부를 예측하는 모델을 구현하였습니다.


## 성능 지표

DKT는 주어진 마지막 문제를 맞았는지(1)와 틀렸는지(0) 분류하는 이진 분류 문제입니다.

<p align="center">
  <img src="image/평가방법.png" width="500">
</p>

성능 지표로 AUROC(Area Under the ROC curve)와 Accuracy를 사용하였습니다.

<p align="center">
  <img src="image/AUCACC.png" width="500">
</p>

AUROC는 ROC 곡선 아래 부분의 넓이를 의미하는데, ROC는 False Positive Rate(x축)과 True Positive Rate(y축)이 이루는 곡선을 의미합니다. AUC가 1이면 모델이 0과 1을 완벽하게 분리해낼 수 있다는 것을 의미하며, 반면에 AUC가 0.5이면 모델이 클래스를 구분하지 못한다는 것을 의미합니다.


## 데이터

<p align="center">
  <img src="image/데이터.png" width="500">
</p>

+ `userID` 사용자의 고유번호 : 총 7,442명의 고유 사용자가 있으며, train/test셋은 이 userID를 기준으로 90/10의 비율로 나누어졌습니다.
+ `assessmentItemID` 문항의 고유번호 : 총 9,454개의 고유 문항이 있습니다.
+ `testId` 시험지의 고유번호 : 총 1,537개의 고유한 시험지가 있습니다.
+ `answerCode` 사용자가 해당 문항을 맞췄는지 여부에 대한 이진 데이터 : 0은 사용자가 해당 문항을 틀린 것, 1은 사용자가 해당 문항을 맞춘 것입니다.
+ `Timestamp` 사용자가 해당문항을 풀기 시작한 시점의 데이터
+ `KnowledgeTag` 중분류 : 총 912개의 고유 태그가 존재합니다.


## EDA & Feature Engineering

1. Feature 세분화 : `assessmentItemID` ⇒ `category`, `test`, `item`</br>
  세분화한 변수들의 정답률을 비교하였을 때, `category`와 `item` 변수는 값에 따라 정답률의 차이를 보여 category 타입의 변수로 가공하였습니다.

2. `KnowledgeTag`를 활용한 `chapter` 변수 생성</br>
  `testId` 별 고유힌 `KnowledgeTag`들을 집합으로 묶었으며, 서로 다른 `testId`를 가지더라도, 두 시험지의 `KnowledgeTag` 집합 사이에 교집합이 존재한다면 두 시험지는 같은 내용을 다루는 시험지라고 판단하여 동일한 `chapter`로 분류하였습니다.
  
3. 학생의 실력 향상 관련 변수 생성</br>
  학습을 통해 실력이 향상될 수 있기 때문에, 학생의 실력을 시간에 따라 나타낼 수 있는 변수들을 추가하였습니다. 현재까지 학생이 푼 모든 문제에 대하여 누적 문제 개수, 누적 정답 개수, 정확도를 변수로 생성하였습니다.

4. 이전 문제를 푸는데 소요된 시간 관련 변수 생성</br>
  `Timestamp` 변수는 문제를 풀기 시작한 시점이므로 다음 문제의 Timestamp에서 현재 Timestamp를 빼 `prev elapsed` 변수를 생성하였습니다. test 데이터의 경우 학생이 푼 마지막 문제이므로 해당 문제를 푸는데 소요된 시간을 알 수 없기 때문에 이전 문제를 푸는데 걸린 시간의 평균(`prev elapsed mean`)을 변수로 생성하였습니다.


