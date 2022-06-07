# Deep Knowledge Tracing (DKT)

## 👋 팀원 소개

|                                                  [김동현](https://github.com/donghyyun)                                                   |                                                                          [임지원](https://github.com/sophi1127)                                                                           |                                                 [이수연](https://github.com/coding-groot)                                                  |                                                                        [진상우](https://github.com/Jin-s-work)                                                                         |                                                                         [심재정](https://github.com/Jaejeong98)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![그림1](https://user-images.githubusercontent.com/61958748/172278471-584ffaf5-ea6d-4e63-ae77-7cac4dbae899.png)| ![그림2](https://user-images.githubusercontent.com/61958748/172278474-f2d54e27-898b-4142-af78-b0e370e43ffc.png)| <img width="140" alt="그림3" src="https://user-images.githubusercontent.com/61958748/172278478-f3bbd8ce-3616-4c37-8fa6-4247e20b469e.png">| ![그림4](https://user-images.githubusercontent.com/61958748/172278482-a591c2e4-f4b7-4edf-a390-9e875c2c4226.png)| ![그림5](https://user-images.githubusercontent.com/61958748/172278489-00773bd6-080f-41ec-b828-24f4dabc5f98.png)|


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
+ `KnowledgeTag`중분류 : 총 912개의 고유 태그가 존재합니다.
