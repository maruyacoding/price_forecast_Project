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

===

## **2. 사과 

