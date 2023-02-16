import sklearn
import pandas as pd
import numpy as np
from sklearn import model_selection # cross-validation score를 가져오기 위함
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier # bagging
from sklearn.tree import DecisionTreeClassifier # 의사 결정 나무
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import Counter # count
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb

""" 시각화 """
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore", UserWarning)

# hyperparameters
param_grid = {'n_estimators': [3, 5, 7, 9],
              'oob_score': [True], # compute out of bag error
              'n_jobs':[-1], 
              'max_depth': [3, 5, 7]
              }

param_dist = {"max_depth": [10,30,50],
              "min_child_weight" : [1,3,6],
              "n_estimators": [200],
              "learning_rate": [0.05, 0.1,0.16]}

class AI_Filter_Level():
    def File_read(self):
        filename = './train_data.csv'
        df = pd.read_csv(filename)    
        df = df.fillna(0)            
        df = df.drop(df[df.suction == 0].index)
        self.array = df.values
        self.df = df
        print(df.head(5))
        
    def File_read_test(self):
        filename = './test_data.csv'
        df = pd.read_csv(filename)    
        df = df.fillna(0)            
        df = df.drop(df[df.suction == 0].index)
        self.array = df.values
        self.test = df
        print(df.head(5))
        
    def Data_set(self):
        # X = self.array[:, 1:4].astype(float)
        # Y = self.array[:, 4]
        X = self.df[['mode', 'pressure', 'rpm']]
        # X.pressure = np.log(self.df.pressure)        
        # X.rpm = np.log(self.df.rpm)        
        # Y = np.log(self.df['suction'])
        Y = self.df.suction
        # Y = self.df.level5
        print(X[:5], Y[:5])
            
        # self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
        self.train_x = X
        self.train_y = Y
        # print('Number of train set:', len(train_x))
        # print('Number of test set:', len(test_x))
        assert len(self.train_x) == len(self.train_y)
        # assert len(self.test_x) == len(self.test_y)
        
    def Data_test_set(self):
        X = self.test[['mode', 'pressure', 'rpm']]
        Y = self.test.suction

        self.test_x = X
        self.test_y = Y
    
    def Trainning(self):
        # 1) 모델 선언 & 2) 여러 모델들을 ensemble: randomforest
        # rf_model = RandomForestClassifier()
        # rf_model = RandomForestRegressor()
        rf_model = xgb.XGBRegressor()
        # self.LR = LinearRegression()        

        # hyperparameter search
        # self.grid_search = GridSearchCV(rf_model, param_grid=param_grid, cv=5, scoring='f1')
        self.grid_search = GridSearchCV(rf_model, param_grid=param_dist, cv = 3,scoring="accuracy", 
                                   verbose=10, n_jobs=-1)
        self.grid_search.fit(self.train_x, self.train_y)
        self.opt_model = self.grid_search.best_estimator_
        # rf_model.fit(self.train_x, self.train_y)
        # self.LR.fit(self.train_x, self.train_y)
        
    def Testing(self):
        test_pred_y = self.opt_model.predict(self.test_x)
        # test_pred_y = self.LR.predict(self.train_x)
        
        mae = np.mean(np.abs(test_pred_y-self.test_y))
        mape = np.mean(np.abs(test_pred_y-self.test_y)/self.test_y)
        
        print('mae:',mae)
        print('mape:',mape)
        
        plt.figure(figsize = (16,8))
        ind = np.argsort(np.array(self.test_y))
        plt.plot(np.array(self.test_y)[ind],label='Real_Value')
        plt.plot(np.array(test_pred_y)[ind],label='Predict_Value')
        plt.legend()
        plt.title('suction power')
        plt.show()
        
    def Check_result_var(self):
        self.opt_model = self.grid_search.best_estimator_
        print(self.opt_model)        
        print(self.opt_model.feature_importances_)
        print(self.opt_model.oob_score_)
        print(self.grid_search.best_params_)      
        
    def Processing(self):
        self.File_read()
        self.Data_set()
        self.Trainning()
        
        self.File_read_test()
        # self.Check_result_var()
        self.Data_test_set()
        self.Testing()       
        
oAI = AI_Filter_Level()
oAI.Processing()


