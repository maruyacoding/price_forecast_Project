# 평가지표 MAPE값 활용

- MAPE(Mean Absolute Percentage Error) : 평균절대오차비율
- MAPE는 실제값과 예측한 값의 차이를 실제값으로 나눈뒤, 데이터 개수만큼 해당값을 더해준 후, 퍼센트로 바꿔줌
- MAE는 Outlier에 취약할 수 있지만 MAPE는 최대 크기가 제한되므로 이를 보완할 수 있음

![image](https://user-images.githubusercontent.com/97514461/200257401-b2a65d7a-36c1-453d-b67c-f7230e33fe38.png)

---

# Train, Validation, Test set

- Time Series Cross Validation(tscv) 활용 -> Stack구조로 앞 단의 데이터 부터 쌓아가면서 model fit
![image](https://user-images.githubusercontent.com/97514461/200257700-fcda4dca-55a9-4c94-a433-c5d4670fd79e.png)

- **Train, Validation set가지고 tscv 진행 -> 최종 모델 선정 -> Test set으로 MAPE값 성능 확인**
