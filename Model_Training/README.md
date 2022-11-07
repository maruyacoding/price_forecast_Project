## **1. baseline model**

feature : SALEDATE(판매일), mean_price(평균단가), year, month, day, weekday

1. 품종별 SALEDATE(판매일) 기준으로 groupby하고 kg당 평균단가(mean_price) feature 추가
2. train : test = 2016.01.01 ~ 2019.09.22 : 2019.09.23 ~ 2020.09.28 (1년치)
3. target feature : 현재 데이터에서 1주일 뒤의 가격 ( '1week' feature 생성)
4. LinearRegression, Lasso, Ridge, LGBM, XGboost, RF (ML model 사용)

---

baseline model 결과 mape 값
(순서대로 LinearRegression, Lasso, Ridge, LGBM, XGboost, RF)
- 사과   : 0.16, 0.16, 0.16, 0.20, 0.31, 0.20
- 양파   : 0.12, 0.12, 0.12, 0.15, 0.16, 0.14
- 대파   : 0.16, 0.16, 0.16, 0.15, 0.14, 0.15
- 무     : 0.18, 0.18, 0.18, 0.18, 0.20, 0.17
- 배추   : 0.25, 0.25, 0.25, 0.19, 0.18, 0.22
- 마늘   : 0.11, 0.11, 0.11, 0.10, 0.11, 0.12
- 건고추 : 0.39, 0.39, 0.39, 0.42, 0.38, 0.42

![image](https://user-images.githubusercontent.com/97514461/200254433-90eebf28-d970-4b3c-9381-078069c6ef36.png)



---


## **2. 사과 data modeling**

데이콘 tscv(time series cross validation), LightGBM Optuna 필사

1. 기본 feature 사용
2. 시계열 데이터 분포 feature 추가
3. 외부변수 제외, 시계열 분포 feature만 사용

- Time series cross validation 5 fold로 진행
- LightGBM Optuna train : test = 8 : 2 로 진행
- LGBM hyper-parameter tuning한 모델로 Predictied, Actual 시각화
- LGBM feature importances 시각화
- Prophet 모델 : 이상치 제거, 휴일 추가, hyper-parameter tuning까지 진행 후 성능 확인


![image](https://user-images.githubusercontent.com/97514461/200254486-0616a698-db32-47cc-aa97-ba0453d457e1.png)



---


## **3. AutoML**

- initialize setup : data_split_shuffle = False, target = '1week', fold_strategy = 'timeseries', fold = 3
- 전품종 AutoML(Pycaret Regression) 돌리고 대략적인 성능 확인

![image](https://user-images.githubusercontent.com/97514461/200254176-40c3deb2-0153-4885-967f-e7ca4cd44612.png)


---


## **4. 다양한 경우의 수로 modeling**
![image](https://user-images.githubusercontent.com/97514461/200255229-c0410cb9-6246-4cfd-b2e3-b9b3db2df500.png)

- RobustSacler 사용
- Time Series Cross Validation 5 fold
![image](https://user-images.githubusercontent.com/97514461/200255492-ecabe0e4-775a-4e8b-a2a4-145e52592f44.png)

---

## **5. 라이브 코딩으로 팀원들과 의논하며 feature selection, engineering 및 modeling 진행**

![image](https://user-images.githubusercontent.com/97514461/200256521-5826c4a9-db4b-4f82-8739-dabbe459a873.png)


