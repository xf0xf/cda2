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


data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(train_x, train_y, test_size=0.25,random_state=13)

lgbm = lgb.LGBMClassifier(silent=False)
param_dist = {"max_depth": np.arange(10,40,5),
            "learning_rate" : [0.02,0.05,0.07],
            "num_leaves": [100,200,300],
            "n_estimators": np.arange(50,100,10)
             }
model = GridSearchCV(lgbm, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="f1", verbose=5)
model.fit(data_train_x,data_train_y)
print('最优参数：', model.best_params_)
model1 = model.best_estimator_
show_accuracy(model1,data_train_x, data_test_x, data_train_y, data_test_y)

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
    

train_x.to_csv('train_x.csv',sep=',',index=False)