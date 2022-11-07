# 함수화

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as grs
# %matplotlib inline
import seaborn as sns

# plt.rc('font', family='Malgun Gothic')
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
# from IPython.display import Image
import plotly.express as px # ploty
import plotly.graph_objects as go

from tqdm import tqdm
import time

#모델
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from lightgbm import LGBMRegressor 
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMClassifier, plot_importance
import lightgbm 

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score

from sklearn.model_selection import TimeSeriesSplit 
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.linear_model import Lars,LassoLars,LassoLarsCV

# Optuna 사용시 주석 제거하고 사용
import optuna
from optuna.integration import XGBoostPruningCallback

pd.options.display.float_format = '{:.2f}'.format

import warnings
warnings.filterwarnings(action='ignore')

import gc, sys
gc.enable() # 자동 가비지 수거 활성화



def split_train_and_test(df, date, week):
    """
    Dataframe에서 train_df, test_df로 나눠주는 함수
    df : 시계열 데이터 프레임
    date : 기준점 날짜
    """
    train = df[df['SALEDATE'] < date]
    test = df[df['SALEDATE'] >= date]
    del train['SALEDATE']
    del test['SALEDATE']
    y_train = train.pop(f'{week}week')
    x_train=train.copy()
    y_test = test.pop(f'{week}week')
    x_test=test.copy()
    return x_train,y_train,x_test,y_test

def eval_model(y_test,pred):
    y_true, y_pred = np.array(y_test), np.array(pred)
    mae = mean_absolute_error(y_true, pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print('mae:',mae,'mape:',mape, 'mse:',mse)
    return mae, mape, mse

#시각화 그래프 
def eval_visul(y_test,pred,title,week):
    y_test=y_test.reset_index()[f'{week}week']
    plt.figure(figsize=(20,5))
    plt.title(title, fontsize = 25)
    plt.plot(y_test, label='true')
    plt.plot(pred, label='pred')
    plt.legend()
    plt.show()
    eval_model(y_test,pred)

def mean_price(df_apple):
    df_apple=df_apple.groupby(['SALEDATE']).mean()
    df_apple['mean_price']=df_apple['TOT_AMT']/df_apple['TOT_QTY']
    #사용된 총금액과 총거래량은 제거
    # df_apple=df_apple.drop(columns=['TOT_AMT','TOT_QTY'])
    df_apple=df_apple.reset_index()
    df_apple = df_apple.round()
    #날짜 컬럼 추가
    # df_apple['year'] = df_apple['SALEDATE'].dt.year
    # df_apple['month'] = df_apple['SALEDATE'].dt.month
    # df_apple['day'] = df_apple['SALEDATE'].dt.day
    # df_apple['weekday'] = df_apple['SALEDATE'].dt.weekday
    return df_apple


#n주일 후 가격을 예측하는 컬럼을 추가 
#df-> 예측기간 가격이 0으로 나오는 값 제외, 따로 변수로 지정
def forcast_week(df,week):
    df[f'{week}week']=0
    
    for index in range(len(df)):
        try:df[f'{week}week'][index] = df['mean_price'][index + (7*week)]
        except:continue
    df = df.drop(df[df[f'{week}week'] == 0].index)
    return df

# time_lag
def train_serise(df_apple, time):
    for lag in range(1, time + 1):
        df_apple[f'p_lag_{lag}'] = -1
        #df_apple[f'q_lag_{lag}'] = -1
        for index in range(lag, len(df_apple)):
            df_apple.loc[index, f'p_lag_{lag}'] = df_apple['mean_price'][index-lag] #1일전, 2일전, ... 가격을 feature로 추가
            #df_apple.loc[index, f'q_lag_{lag}'] = df_apple['TOT_QTY'][index-lag] #1일전, 2일전, ... 거래량을 feature로 추가
    return df_apple


#요일컬럼 추가 -> 원 핫 인코딩
def weekday(df_test):
    weekday=['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    df_test['요일']=0
    for i in range(len(df_test)):
        df_test['요일'][i]=weekday[df_test['SALEDATE'][i].weekday()]
    df_test = pd.concat([df_test, pd.get_dummies(df_test['요일'])], axis=1)
    df_test =df_test.drop(columns='요일')
    return df_test

#시계열 분해 잔차활용
def resid(df):
    date_resid=STL(df[['SALEDATE','mean_price']].set_index('SALEDATE'), period=12)
    df['resid']=date_resid.fit().resid.values
    return df

#시계열 분해 트렌드활용
def trend(df):
    date_resid=STL(df[['SALEDATE','mean_price']].set_index('SALEDATE'), period=12)
    df['trend']=date_resid.fit().trend.values
    return df

#시계열 분해 시즌활용
def season(df):
    date_resid=STL(df[['SALEDATE','mean_price']].set_index('SALEDATE'), period=12)
    df['season']=date_resid.fit().seasonal.values
    return df

def add_price(df_pum):
  df_pum["mean_price_5"]= df_pum["mean_price"].rolling(5).mean().shift(1) # 지난 5일간 평균 가격
  df_pum["mean_price_30"]= df_pum["mean_price"].rolling(21).mean().shift(1) # 지난달 평균 가격 
  df_pum["ratio_mean_price_5_30"] = df_pum["mean_price_5"]/df_pum["mean_price_30"] # 5일간/지난달 비율 
  return df_pum

def add_std_price(df_pum):
  df_pum["std_price_5"]= df_pum["mean_price"].rolling(5).std().shift(1) # 지난 5일간 평균 가격 표준편차
  df_pum["std_price_30"]= df_pum["mean_price"].rolling(21).std().shift(1) # 지난달 평균 가격 표준편차 
  df_pum["ratio_std_price_5_30"] = df_pum["std_price_5"] / df_pum["std_price_30"]
  return df_pum




# 7일 예측 기준
def select_pum2(df, pum, time):
    df_apple = df[df['PUM_NM']==pum]
    df_apple = df_apple[['SALEDATE', 'PUM_NM', 'KIND_NM', 'SAN_NM', 'TOT_AMT', 'TOT_QTY']]
    # 경제지표 외부변수 넣어서 확인
    # df_apple = df_apple[['SALEDATE', 'PUM_NM', 'KIND_NM', 'SAN_NM', 'TOT_AMT', 'TOT_QTY', 'CD', 'Exchange_Rate']]
    df_apple['mean_price'] = df_apple['TOT_AMT'] / df_apple['TOT_QTY']
    df_apple = mean_price(df_apple)
    # time_lag
    df_apple = train_serise(df_apple, time)
    # 시계열 관련 feature
    df_apple = resid(df_apple)
    df_apple = trend(df_apple)
    df_apple = season(df_apple)
    # 가격 관련 이동평균선
    df_apple = add_price(df_apple)
    df_apple = add_std_price(df_apple)
    df_apple = df_apple.dropna(axis=0)
    df_apple = forcast_week(df_apple, 1) # 1주일 뒤의 가격 예측
    df_apple = df_apple.reset_index(drop = True)
    return df_apple


def pre_2(df, pum, time) : 
    df_pum=select_pum2(df, pum, time)

    # train, test split
    X_train,y_train,X_test,y_test=split_train_and_test(df_pum,'2019-09-22',1)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    x_columns = list(df_pum.columns)
    x_columns.remove('SALEDATE')
    x_columns.remove('1week')
    x_columns

    scaler = StandardScaler()
    # scaler = RobustScaler()

    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, index= range(X_train.shape[0]), columns = x_columns) # feature importances 그래프를 위해 dataframe화
    X_test = scaler.transform(X_test) 
    X_test = pd.DataFrame(X_test, index= range(X_test.shape[0]), columns = x_columns)

    return X_train, X_test, y_train, y_test


def MAPE(y, y_pred):
    mape = mean_absolute_percentage_error(y, y_pred)
    return mape

def mape_cv2(model, df, pum, time):
    train_X, test_X, train_y, test_y = pre_2(df, pum, time)
    # cv별로 학습하는 함수
    tscv = TimeSeriesSplit(n_splits=5)
    mape_list = []
    model_name = model.__class__.__name__
    for _, (train_index, test_index) in tqdm(enumerate(tscv.split(train_X)), desc=f'{model_name} Cross Validations...', total=10):
        X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]
        print("First train_day: {}\t Last train_day: {}".format(X_train[:1].index[0], X_train[-1:].index[0]))
        print("First val_day: {}\t Last val_day: {}".format(X_test[:1].index[0], X_test[-1:].index[0]))
        print('-' * 40)
        clf = model.fit(X_train, y_train)
        pred = clf.predict(X_test)
        mape = MAPE(y_test, pred) 
        mape_list.append(mape)
    return model_name, mape_list


def print_mape_score2(model, df, pum, time):
    # cv별 프린팅, 평균 저장
    model_name, score = mape_cv2(model, df, pum, time)
    for i, r in enumerate(score, start=1):
        print(f'{i} FOLDS: {model_name} MAPE: {r:.4f}')
    print(f'\n{model_name} mean MAPE: {np.mean(score):.4f}')
    print('='*40)
    return model_name, np.mean(score)

# 기본 모델
reg2 = LinearRegression()
ridge2 = Ridge()
lasso2 = Lasso()
Enet2 = ElasticNet()
DTree2 = DecisionTreeRegressor()
rf2 = RandomForestRegressor()
model_xgb2 = XGBRegressor()
model_lgb2 = LGBMRegressor()

def modeling2(df, pum, time, x_columns) :
    models = []
    scores = []
    # 기본 모델로
    for model in [reg2, ridge2, lasso2, Enet2, DTree2, rf2, model_xgb2, model_lgb2]:
        model_name, mean_score = print_mape_score2(model, df, pum, time)
        models.append(model_name)
        scores.append(mean_score)
    result_df = pd.DataFrame({'Model': models, 'Score': scores}).reset_index(drop=True)
    display(result_df)

    f, ax = plt.subplots(figsize=(10, 6))
    plt.xticks(rotation = 90)
    sns.barplot(x=result_df['Model'], y=result_df['Score'])
    plt.xlabel('Models', fontsize=15)
    plt.ylabel('Model Performance', fontsize=15)
    plt.ylim(0, 1)
    plt.title(f'{pum} 7일 예측 MAPE', fontsize=15)
    plt.show()

    # LightGBM feature importances 시각화
    if model == model_lgb2 :
        plt.figure(figsize = (10,8))
        plt.barh(x_columns, model.feature_importances_)
        plt.title(f'{pum} LightGBM feature importances', fontsize = 15)
        plt.show()
