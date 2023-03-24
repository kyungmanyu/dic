import sklearn
import pandas as pd
import numpy as np
# from sklearn import model_selection # cross-validation score를 가져오기 위함
from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import BaggingClassifier # bagging
# from sklearn.tree import DecisionTreeClassifier # 의사 결정 나무
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingRegressor, StackingClassifier
from collections import Counter # count
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
import lightgbm as lgb

""" 시각화 """
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore", UserWarning)

# hyperparameters
param_rf = {'n_estimators': [50,100,150,200],
              'oob_score': [True], # compute out of bag error
              'n_jobs':[-1], 
              'max_depth': [25, 50, 75] }

param_xgb = {"max_depth": [25,50,75, 100],
              "min_child_weight" : [1,3,6],
              "n_estimators": [100, 200, 300], 
              "learning_rate": [0.01, 0.05, 0.1]  }

param_lgb = { "objective":['multiclass'], # multiclass, regression
              "max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [100,200,300]     }

train_file = './train_data.csv'
test_file = './test_data.csv'

Ensenmble_model = {'rf': [RandomForestClassifier(),RandomForestRegressor()], \
                   'xgb': [xgb.XGBClassifier(),xgb.XGBRegressor()],\
                   'lgb': [lgb.LGBMClassifier(), lgb.LGBMRegressor()]}

Ensenmble_param = {'rf':[param_rf, param_rf], \
                   'xgb':[param_xgb, param_xgb],\
                   'lgb':[param_lgb, param_lgb]}

trainning = {'Classifier':0, 'Regressor':1}   

class AI_Filter_Level():
    train_x = 0
    train_y = 0
    best_acc = 0
    def File_read(self, type):
        if type == 'train':
            filename = train_file
        else :
            filename = test_file
            
        df = pd.read_csv(filename)    
        df = df.fillna(0)            
        df = df.drop(df[df.suction == 0].index)
               
        return df
        
    def Data_set(self, df, type, split = 0):       
        # df.pressure = np.log(df.pressure)
        # df.rpm = np.log(df.rpm)
        # df = df[df['mode'] == 2.0]
        # df = df.drop(df[df.orifice == 30].index)
        
        # x 변수 선택
        # X = df[['mode', 'pressure', 'rpm', 'orifice']]
        X = df[['mode', 'pressure', 'rpm']]
        # X = df[['pressure','rpm']]
        
        if type == 'suction':
            Y = df.suction
        elif type == 'orifice':    
            Y = df.orifice
        elif type == 'level':
            Y = df.level5
            
        # print(X[:5], Y[:5])
        
        if split != 0:  
            train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = split, random_state=0)
            return train_x, test_x, train_y, test_y
        else:
            return X, Y            
                      
    def Trainning_ensenmble(self, model, type):
        model_ensen = Ensenmble_model[model][trainning[type]]
        parm = Ensenmble_param[model][trainning[type]]
        
        # hyperparameter search                
        if type == 'Regressor': self.grid_search = GridSearchCV(model_ensen, param_grid=parm, cv = 25)
        else:self.grid_search = GridSearchCV(model_ensen, param_grid=parm, cv=25, scoring = 'accuracy')
        
        self.grid_search.fit(self.train_x, self.train_y)        
        self.opt_model = self.grid_search.best_estimator_   
        
        # self.opt_model.fit(self.train_x, self.train_y)
        print(self.opt_model)
        
    def Trainning_MLP(self, player = (4,4,2), type = 'Classifier'):   
    # mode, pressure, rpm
    #(5,4,3)(8,7,6)(12,9,7)(13,8,3)    
    #(4,4,2)_classifier(5,4),(6,5)
    # pressure, rpm, mode = 2 fix
    #(5,2,2)(7,3,1)
        
        layer = player
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_x)
        self.train_x = self.scaler.transform(self.train_x)
        
        if type == 'Classifier':
            self.reg_mlp = MLPClassifier(activation='relu', alpha=0.001, batch_size=10,
                        hidden_layer_sizes=layer, max_iter=10000, # 16,9,3
                        solver='adam', verbose = False, random_state = 2023)

        elif type == 'Regressor':
            self.reg_mlp = MLPRegressor(activation='relu', alpha=0.001, batch_size=10,
                        hidden_layer_sizes=layer, max_iter=10000, # 16,9,3
                        solver='adam', verbose = False, random_state = 2023)
        
        self.reg_mlp.fit(self.train_x, self.train_y)     
    
        return self.reg_mlp
        train_loss_values = self.reg_mlp.loss_curve_        
        # print(self.reg_mlp.coefs_)
        
        plt.figure(figsize=(20,10))
        plt.plot(train_loss_values,label='Train Loss')

        plt.legend(fontsize=20)
        plt.title("Learning Curve of trained MLP Regressor", fontsize=18)
        plt.show()        
        
    def Testing(self, model, type, view = True):
        if model == 'ensemble':
            test_pred_y = self.opt_model.predict(self.test_x)
            train_pred_y = self.opt_model.predict(self.train_x)
        elif model == 'MLP':
            self.test_x = self.scaler.transform(self.test_x)
            test_pred_y = self.reg_mlp.predict(self.test_x)
            train_pred_y = self.reg_mlp.predict(self.train_x)
            
        if type == 'Classifier':
            acc_score_test =accuracy_score(y_true=self.test_y, y_pred=test_pred_y)
            print('acc_score_test',acc_score_test)
            acc_score_train =accuracy_score(y_true=self.train_y, y_pred=train_pred_y)            
            print('acc_score_train',acc_score_train)   
            if(self.best_acc < acc_score_test):
                self.best_acc = acc_score_test      
               
        elif type == 'Regressor':
            mae = np.mean(np.abs(test_pred_y-self.test_y))
            print('test mae:',mae)
            mae = np.mean(np.abs(train_pred_y-self.train_y))
            print('train mae:',mae)
            # mape = np.mean(np.abs(test_pred_y-self.test_y)/self.test_y)
            # print('mape:',mape)        
        
        if view == False : return
        
        plt.figure(figsize = (16,8))
        ind = np.argsort(np.array(self.test_y))
        plt.plot(np.array(self.test_y)[ind],label='Real_Value')
        plt.plot(np.array(test_pred_y)[ind],label='Predict_Value')
        plt.legend()
        plt.title('suction power')
        plt.show()

    def Optimal_MLP(self, type = 'Classifier'):        
        # for d3 in range(3,30):
            for d2 in range(2,30):
                # if d2 > d3: continue
                for d1 in range(1,30):       
                    if d1 > d2: continue
                    # layer = (d3,d2,d1)
                    layer = (d2,d1)
                    print(layer)
                    self.Trainning_MLP(layer, type)    
                    self.Testing('MLP', type, False)      

    def stacking(self, type):        
        if type == 'Regressor' :
            #Stacking할 모델들 list로 정의
            model_list = [('rf',RandomForestRegressor(max_depth=25, n_estimators=150, n_jobs=-1, oob_score=True)),
                          ('xgb',xgb.XGBRegressor(max_depth=25, n_estimators=200, learning_rate=0.1,  min_child_weight=6)),
                          ('lgb',lgb.LGBMRegressor(max_depth=25, n_estimators=300, num_leaves=300, objective='regression'))]
            #Final model 정의
            model_final = LinearRegression()
            #Stacking 모델 정의
            model = StackingRegressor(estimators=model_list,final_estimator=model_final,cv=25)
        else :            
            model_list = [('rf',RandomForestClassifier(max_depth=50, n_estimators=200, n_jobs=-1, oob_score=True)),  # mode, pressure, rpm                          
                           ('xgb',xgb.XGBClassifier(max_depth=25, n_estimators=200, learning_rate=0.05,  min_child_weight=1)),
                           ('lgb',lgb.LGBMClassifier(learning_rate=0.05, max_depth=25, num_leaves=300, objective='multiclass'))]            
            model_final = LogisticRegression()
            model = StackingClassifier(estimators=model_list,final_estimator=model_final,cv=5)       
        
        #Stacking 모델 학습
        model.fit(self.train_x,self.train_y)
        
        self.opt_model = model          
        
    def save_data(self):
        test_pred_y = self.opt_model.predict(self.test_x)      
        train_pred_y = self.opt_model.predict(self.train_x)     
        
    def leave_one_out(self,df,type,trainning_type):
        
        
        X = df[['mode', 'pressure', 'rpm']]
        # X = df[['pressure','rpm']]
        print('X data',X.head())
        if type == 'suction':
            Y = df.suction
        elif type == 'orifice':    
            Y = df.orifice
        elif type == 'level':
            Y = df.level5
        
        # self.train_x, self.train_y 99
        # self.test_x, self.test_y 1
        print('lenth of data',len(X))
        for i in range (0, len(X)):
            self.test_x = X.loc[[i],:]
            self.test_x  = self.test_x.head(1)
            self.test_y = Y.loc[[i]]
            self.test_y  = self.test_y.head(1)
            
            self.train_x = X.drop(X.index[i],axis = 0)
            self.train_y = Y.drop(Y.index[i],axis = 0)
        
            self.Trainning_ensenmble('rf', trainning_type)   
            self.Testing('ensemble', trainning_type, False)  
            
        print('best acc by using leave one out', self.best_acc)
 
        
        
    def Processing(self):
        trainning_target = 'level' # level, suction
        
        if trainning_target == 'level': trainning_type = 'Classifier'
        else : trainning_type = 'Regressor'        
        
        df_train = self.File_read('train')
        df_test = self.File_read('test')
        
        df_data = pd.concat([df_train, df_test], axis=0)
        # print(df_data.shape)
        
        self.leave_one_out(df_data,trainning_target,trainning_type)
        
        # self.train_x, self.train_y = self.Data_set(df_train, trainning_target, 0)
        # self.test_x, self.test_y = self.Data_set(df_test, trainning_target, 0)
        
        # # self.Trainning_ensenmble('rf', trainning_type) # model : rf, xgb , lgb
        # # self.stacking(trainning_type)
        # # self.Trainning_MLP(trainning_type)        
        # self.Optimal_MLP()              
        
        # # self.Testing('ensemble', trainning_type)  
        # # self.Testing('MLP', trainning_type)               
        
        print('finish')
     
oAI = AI_Filter_Level()
oAI.Processing()


