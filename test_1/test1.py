# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:22:56 2019

@author: xfxf
"""

import pandas as pd
import numpy as np
import scipy as scp
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split



print(os.getcwd())
os.chdir('D:\\work\\sky_drive\\datamining\\cda2\\test_1\\testdata1')
test = pd.read_csv("test30.csv",na_values= ['nan','?'])
train = pd.read_csv("training30.csv",index_col=0,na_values= ['nan','?'])

train_y = train['Purchase']
train_x0 = train.drop(['Purchase'],axis=1)

#automl

train_y.value_counts()


def fe(df):
    def num_missing(x):  return sum(x.isnull()) 
    df['num_null'] = train.apply(num_missing, axis=1)
    #增加字段是否为空标识
    na_list = ['score','gender','age','using_time','balance','usage','card','Active','salary']
    na_list_name = [a+'_isna' for a in na_list]
    df[na_list_name] = df[na_list].isnull()
    df[na_list_name] = df[na_list_name].astype(int)
    
    #连续变量缺失值填充
    col_input = ['score','age','using_time','balance','usage','card','Active','salary']
    input_num = df[col_input].median()
    for col in col_input:
        df[col] = df[col].fillna(input_num[col])
    
    #离散变量缺失值填充
    
    #one-hot
    col_onehot = ['area','gender']
    df = pd.get_dummies(df,columns=col_onehot)
    
    #处理极值
    train['salary'][train['salary']<1796] = 1796
    
    #标准化
    col_standar = ['balance','salary']
    scaler = StandardScaler()
    df[col_standar] = scaler.fit_transform(df[col_standar])

    return df

train_x = fe(train_x0)


from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb


data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(train_x, train_y, test_size=0.25,random_state=59)

def show_accuracy(model,x_train,x_test,y_train,y_test):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    from sklearn.metrics import roc_curve
    from sklearn import metrics
    
    y_train_pre = model.predict(x_train)
    probs_train = model.predict_proba(x_train)[:,1]
    print('训练集正确率(accuracy_score)：', accuracy_score(y_train, y_train_pre))
    print('训练集召回率(recall_score)：', recall_score(y_train, y_train_pre))
    print('训练集精确度：', precision_score(y_train, y_train_pre))
    print('训练集F1 值：', f1_score(y_train, y_train_pre))
    fpr,tpr,thresholds = roc_curve(y_train,probs_train)
    print('训练集AUC：',metrics.auc(fpr,tpr))
    
    y_test_pre = model.predict(x_test)
    probs_test = model.predict_proba(x_test)[:,1]
    print('测试集正确率(accuracy_score)：', accuracy_score(y_test, y_test_pre))
    print('测试集召回率(recall_score)：', recall_score(y_test, y_test_pre))
    print('测试集精确度：', precision_score(y_test, y_test_pre))
    print('测试集F1 值：', f1_score(y_test, y_test_pre))
    fpr,tpr,thresholds = roc_curve(y_test,probs_test)
    print('测试集AUC：',metrics.auc(fpr,tpr))

#####自定义调参
lgbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'auc',
                         verbose = 0,
                         learning_rate = 0.1,
                         num_leaves = 35,
                         feature_fraction=0.8,
                         bagging_fraction= 0.9,
                         bagging_freq= 8,
                         lambda_l1= 0.6,
                         lambda_l2= 0)
param_dist = {"max_depth": np.arange(10,40,5),
            "learning_rate" : [0.02,0.05,0.07],
            "num_leaves": [100,200,300],
            "n_estimators": np.arange(50,100,10)
             }
model = GridSearchCV(lgbm, n_jobs=-1, param_grid=param_dist, cv = 5, scoring="f1", verbose=5)
model.fit(data_train_x,data_train_y)
print('最优参数：', model.best_params_)
model1 = model.best_estimator_
show_accuracy(model1,data_train_x, data_test_x, data_train_y, data_test_y)

train_x.to_csv('train_x.csv',sep=',',index=False)

####复杂调参
parameters = {
              'max_depth': [15, 20, 25, 30, 35],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_freq': [2, 4, 5, 6, 8],
              'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              'lambda_l2': [0, 10, 15, 35, 40],
              'cat_smooth': [1, 10, 15, 20, 35]
}
gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'auc',
                         verbose = 0,
                         learning_rate = 0.01,
                         num_leaves = 35,
                         feature_fraction=0.8,
                         bagging_fraction= 0.9,
                         bagging_freq= 8,
                         lambda_l1= 0.6,
                         lambda_l2= 0)
# 有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
gsearch.fit(data_train_x,data_train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

####多阶段调参
##一、
params = {    'boosting_type': 'gbdt', 
    'objective': 'binary', 
    'learning_rate': 0.05, 
    'num_leaves': 50, 
    'max_depth': 6,    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    }
data_train = lgb.Dataset(data_train_x,data_train_y, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])


###二、
model_lgb = lgb.LGBMClassifier(objective='binary',
                              num_leaves=50,
                              learning_rate=0.05, 
                              n_estimators=112, 
                              max_depth=6,
                              metric='auc', 
                              bagging_fraction = 0.8,
                              feature_fraction = 0.8)
params_test1 = {'max_depth': range(3,8,1),
                'num_leaves':range(50, 170, 10)}
gsearch1 = GridSearchCV(estimator=model_lgb, 
                        param_grid=params_test1, 
                        scoring='f1', 
                        cv=5, verbose=1, n_jobs=-1)
gsearch1.fit(data_train_x,data_train_y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


###三、
params_test3={'min_child_samples': np.arange(18, 22),
              #'min_child_weight':np.arange(0.001, 0.005, 0.001)
              }
model_lgb = lgb.LGBMClassifier(objective='binary',
                               num_leaves=80,
                               learning_rate=0.05, 
                               n_estimators=112, 
                               max_depth=7, 
                               metric='auc', 
                               bagging_fraction = 0.8, 
                               feature_fraction = 0.8)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
gsearch3.fit(data_train_x,data_train_y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
    
    
1111    