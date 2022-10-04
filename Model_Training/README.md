**baseline model**

feature : 농산물 도매 거래 데이터, 기상데이터, 유가데이터, 재배면적, 소비자 물가지수 , 날짜데이터(year, month, day, weekday)

1. 각 품종의 국내 주산지 거래량 top1 기준으로 기상데이터 merge
2. 품종별 SALEDATE(판매일) 기준으로 groupby하고 다른 feature는 mean값으로 진행
3. train : test = 2016-01-01 ~ 2020-08-31 : 2020-09 (한달)
4. StandardScaler 사용
5. LinearRegression, Lasso, Ridge, LGBM, XGboost, RF 모델 사용


baseline model 결과 mape 값
(순서대로 LinearRegression, Lasso, Ridge, LGBM, XGboost, RF)
- 사과   : 0.28, 0.30, 0.28, 0.21, 0.20, 0.19
- 양파   : 0.25, 0.31, 0.25, 0.24, 0.25, 0.20
- 대파   : 0.44, 0.49, 0.45, 0.35, 0.33, 0.37
- 무     : 0.56, 0.60, 0.57, 0.43, 0.41, 0.52
- 배추   : 0.23, 0.24, 0.23, 0.21, 0.37, 0.15
- 마늘   : 0.67, 0.67, 0.67, 0.92, 0.82, 0.99
- 건고추 : 0.38, 0.38, 0.38, 0.21, 0.22, 0.24

