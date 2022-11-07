## Final_model
- Final model : Linear Regression model
- Feature Selection / Feature Engineering : time_lag 7일, resid, trend, season, 이동평균선 
![image](https://user-images.githubusercontent.com/97514461/200261651-c5c9bac6-9e9a-4096-9b06-04fe4e1dea92.png)

- target feature : 현재 데이터로 1주일 뒤 가격 예측
![image (1)](https://user-images.githubusercontent.com/97514461/200264158-a40844b5-fe74-41ab-875e-73c6eaed3ea0.png)

- Train, test = 8 : 2(약 1년)
- Standard Scaler 사용 
  - MinMax, Robust도 사용해봤으나 성능에 차이가 없음 -> 보편적으로 많이 사용하는 Standard Scaler 사용


--- 
### 최종 모델로 1주, 2주, 4주 예측 진행
#### 1주 예측 / 사과 Linear Regression 예측값, 실제값 시각화
![image](https://user-images.githubusercontent.com/97514461/200261194-e2608d01-0636-43b5-8507-3f68530358a3.png)

#### 2주 예측 / 사과 Linear Regression 예측값, 실제값 시각화
![image](https://user-images.githubusercontent.com/97514461/200262366-b6305e8e-bec9-4bb5-b402-473b047231e9.png)


#### 4주 예측 / 사과 Linear Regression 예측값, 실제값 시각화
![image](https://user-images.githubusercontent.com/97514461/200262420-c1a7cd60-081e-4b4a-b2e7-fa3613c7ebc8.png)


#### 1, 2, 4주 MAPE값 비교
![image](https://user-images.githubusercontent.com/97514461/200262689-064fd9fa-b181-42c0-a9eb-49b63dc26985.png)
