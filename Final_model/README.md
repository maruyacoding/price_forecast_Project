## Final_model
- Final model : Linear Regression model
- Feature Selection / Feature Engineering : time_lag 7일, resid, trend, season, 이동평균선 
- Train, test = 8 : 2(약 1년)
- Standard Scaler 사용 
  - MinMax, Robust도 사용해봤으나 성능에 차이가 없음 -> 보편적으로 많이 사용하는 Standard Scaler 사용
![image](https://user-images.githubusercontent.com/97514461/200261194-e2608d01-0636-43b5-8507-3f68530358a3.png)
![image](https://user-images.githubusercontent.com/97514461/200261651-c5c9bac6-9e9a-4096-9b06-04fe4e1dea92.png)
