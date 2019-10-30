# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:22:56 2019

@author: xfxf
"""
0.先样本均衡，后测试
1.先分数据，再样本均衡
3.多模型综合
4.imblearn
5.组合特征
6.greadsearch vs 

########## 一、导入 ##########

import pandas as pd
import numpy as np
import scipy as scp
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



print(os.getcwd())
#os.chdir('D:\\work\\sky_drive\\git\\cda2\\testdata1')
os.chdir('D:\\py_project\\cda2\\testdata1')

test = pd.read_csv("test30.csv",na_values= ['nan','?'])
train = pd.read_csv("training30.csv",na_values= ['nan','?'])
test.columns = train.columns

########## 二、特征工程 ##########

#automl
na_list = ['score','gender','age','using_time','balance','usage','card','Active','salary']
col_number_input = ['score','age','using_time','balance','usage','card','Active','salary']
input_num = train[col_number_input].median()
col_class_input = ['age']
input_class = train[col_class_input].mode()
col_onehot = ['area','gender']
col_standar = ['balance','salary']

def fe(data):
    df = data.copy()
    #缺失个数
    def num_missing(x):  return sum(x.isnull()) 
    df['num_null'] = df.apply(num_missing, axis=1)
    
    #增加字段是否为空标识
    na_list_name = [a+'_isna' for a in na_list]
    df[na_list_name] = df[na_list].isnull()
    df[na_list_name] = df[na_list_name].astype(int)
    
    #连续变量缺失值填充
    for col in col_number_input:
        df[col] = df[col].fillna(input_num[col])
    
    #离散变量缺失值填充
    for col in col_class_input:
        df[col] = df[col].fillna(input_class[col])
    
    #one-hot
    df = pd.get_dummies(df,columns=col_onehot)
    
    #处理极值
    df['salary'][df['salary']<1796] = 1796
    
    #标准化
    scaler = StandardScaler()
    df_col_stander = df[col_standar].copy()
    df[col_standar] = scaler.fit_transform(df_col_stander)

    return df

train_fe = fe(train)
test_fe = fe(test)
#train_x = fe(train)
#train_x.to_csv('train_x.csv',sep=',',index=True)

########## 三、数据变换 ##########

###（一）先样本均衡，再分数据

#1.样本均衡
def sample_banlance(train_0,train_1):
    n = min(len(train_0),len(train_1))
    sample_train_0 = train_0.sample(n)
    sample_train_1 = train_1.sample(n)
    data = pd.concat([sample_train_0,sample_train_1])
    return data

train_fe_banlance = sample_banlance(train_fe[train_fe['Purchase']==1],train_fe[train_fe['Purchase']==0])
train_fe_banlance['Purchase'].value_counts()

#2.数据拆分
train_feban_y = train_fe_banlance['Purchase']
train_feban_x0 = train_fe_banlance.drop(['Purchase','ID'],axis=1)

data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(train_feban_x0, train_feban_y, test_size=0.25,random_state=59)
data_train_y.value_counts()

###（二）先分数据，再做样本均衡

#1.数据拆分
train_feban_y = train_fe['Purchase']
train_feban_x0 = train_fe.drop(['Purchase','ID'],axis=1)
data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(train_feban_x0,train_feban_y , test_size=0.25,random_state=59)

#2.样本均衡
#data_train_y.value_counts()
#
#data_train_x['Purchase'] = data_train_y
#train_fe_banlance = sample_banlance(data_train_x[data_train_x['Purchase']==1],data_train_x[data_train_x['Purchase']==0])
#data_train_y = train_fe_banlance['Purchase']
#data_train_x = train_fe_banlance.drop(['Purchase'],axis=1)
#
#data_train_y.value_counts()

########## 四、模型训练 ##########

from sklearn.linear_model.logistic import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

def show_accuracy(model,x_train=data_train_x,x_test=data_test_x,y_train=data_train_y,y_test=data_test_y):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    from sklearn.metrics import roc_curve
    from sklearn import metrics
    
    y_train_pre = model.predict(x_train)
    probs_train = model.predict_proba(x_train)[:,1]
    fpr,tpr,thresholds = roc_curve(y_train,probs_train)
    y_test_pre = model.predict(x_test)
    probs_test = model.predict_proba(x_test)[:,1]
    fpr,tpr,thresholds = roc_curve(y_test,probs_test)

    model_accu = {'train_accuracy_score':round(accuracy_score(y_train, y_train_pre),4),
                  'train_recall_score':round(recall_score(y_train, y_train_pre),4),
                  'train_precision_score':round(precision_score(y_train, y_train_pre),4),
                  'train_f1_score':round(f1_score(y_train, y_train_pre),4),
                  'train_auc':round(metrics.auc(fpr,tpr),4),
                  'test_accuracy_score':round(accuracy_score(y_test, y_test_pre),4),
                  'test_recall_score':round(recall_score(y_test, y_test_pre),4),
                  'test_precision_score':round(precision_score(y_test, y_test_pre),4),
                  'test_f1_score':round(f1_score(y_test, y_test_pre),4),
                  'test_auc':round(metrics.auc(fpr,tpr),4)
                  }
    #print(pd.Series(model_accu))
    return pd.Series(model_accu)

model_dict = {}

#逻辑回归
lr = LogisticRegression(penalty='l2') 
#设置样本为非平衡样本  设置正负样本权重lr = LogisticRegression(class_weight={0:0.3,1:0.7}) 
param = {'C':np.arange(0.5,2,0.1),
         'penalty':['l1','l2']
        }
gsearch_lr = GridSearchCV(lr, param_grid=param, cv=4,scoring='f1',n_jobs=-1,verbose=5)
gsearch_lr.fit(data_train_x, data_train_y)
model_lr = gsearch_lr.best_estimator_
model_dict['model_lr'] = model_lr
gsearch_lr.best_params_,gsearch_lr.best_score_

#svm
clf=svm.SVC(C=0.8, kernel='rbf')
param_test = {
 'C':np.arange(0.1,2.0,0.1),
 'kernel':['linear','poly','rbf','sigmoid','precomputed']}
gsearch_svm = GridSearchCV(clf,param_grid = param_test,scoring='f1',n_jobs=-1,cv=4,verbose=5)
gsearch_svm.fit(data_train_x, data_train_y)
model_svm = gsearch_svm.best_estimator_
model_dict['model_svm'] = model_svm
gsearch_svm.best_params_, gsearch_svm.best_score_


#随机森林
print('#'*10+'随机森林调参'+'#'*10)
rf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=7, min_samples_leaf=11, min_samples_split=30, max_features=0.5,n_jobs=-1)
param = {'n_estimators':np.arange(50,200,25),
         'max_features':np.arange(2, 8),
        'max_depth':np.arange(2,10,2), 
        #'min_samples_leaf':np.arange(2,10,2),
        #'min_samples_split':np.arange(2, 53, 10)
		'class_weight':['balanced', None]
        }
gsearch_rf = GridSearchCV(rf, param_grid=param, cv=4,scoring='f1',n_jobs=-1,verbose=1)
gsearch_rf.fit(data_train_x, data_train_y)
model_rf = gsearch_rf.best_estimator_
model_dict['model_rf'] = model_rf
gsearch_rf.best_params_, gsearch_rf.best_score_

rf_model_resule = show_accuracy(model1,data_train_x, data_test_x, data_train_y, data_test_y)


#xgboost
gbm = xgb.XGBClassifier(learning_rate =0.1,n_estimators=100,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=13)
param = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(3,10,1),
 'gamma':np.arange(0.1,0.8,0.1),
 'subsample':np.arange(0.1,1,0.1),
 'colsample_bytree':np.arange(0.1,1,0.1)}
gsearch_xgb = GridSearchCV(gbm,param_grid = param,scoring='f1',n_jobs=-1,cv=4,verbose=5)
gsearch_xgb.fit(data_train_x, data_train_y)
print(gsearch_xgb.best_params_,gsearch_xgb.best_score_)
model_xgb = gsearch_xgb.best_estimator_
model_dict['model_xgb'] = model_xgb




#lightgbm 自定义调参
lgbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'auc',
                         #is_unbalance=True,
                         learning_rate = 0.01,
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
gsearch_lgb = GridSearchCV(lgbm, n_jobs=-1, param_grid=param_dist, cv = 5, scoring="f1", verbose=5)
gsearch_lgb.fit(data_train_x,data_train_y)
print(gsearch_lgb.best_params_,gsearch_lgb.best_score_)
model_lgb = gsearch_lgb.best_estimator_
model_dict['model_lgb'] = model_lgb


show_accuracy(model1,data_train_x, data_test_x, data_train_y, data_test_y)

#模型组合
1.定义函数，参数选定使用哪些模型；
2.通过网格搜索确定模型最优参数；
3.输出模型训练集测试集得分情况，dataframe，多个模型结果横向对比
4.输出模型
5.模型结果计算
def models_score(model_dict):
    result_score = pd.DataFrame()
    for name,model in model_dict.items():
        result_score[name] = show_accuracy(model)
    return result_score

print(models_score(model_dict))
#模型综合
1.输入多个模型，输入数据
2.使用模型分别预测出结果，并输出综合后的结果
3.综合的方式包括投票（软、硬）

def models_predict(model_dict,test_data):
    result = pd.DataFrame()
    for name, model in model_dict.items():
        result[name] = model.predict(test_data)
    result['volte_soft'] = result.apply(lambda x : 0 if np.sum(x)<(len(x)/2) else 1,axis=1)
    result['volte_hard'] = result.apply(lambda x : 0 if np.sum(x)==0 else 1,axis=1)
    return result

test_fe = test_fe.drop(['Purchase','ID'],axis=1)
result = models_predict(model_dict,test_fe)

'''
1.多种模型结果最后训练分别提交
2.搜索出来的最佳模型直接作为成熟模型；
3.将模型参数抄出来重新单独训练的模型；（在全量训练数据下进一步训练）
4.将模型参数抄出来重新单独训练的模型；（和全量训练数据构建平衡样本训练数据进一步训练）
'''
#3.数据
train_feban_y = train_fe['Purchase']
train_feban_x0 = train_fe.drop(['Purchase','ID'],axis=1)
#4.数据
train_fe_banlance = sample_banlance(train_fe[train_fe['Purchase']==1],train_fe[train_fe['Purchase']==0])
train_fe_banlance['Purchase'].value_counts()
train_feban_y = train_fe['Purchase']
train_feban_x0 = train_fe.drop(['Purchase','ID'],axis=1)


########## 五、预测 ##########
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
# rfc_submission.to_csv('rfc_submission.csv', index=False)


####复杂调参
parameters = {
        #'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
        #'n_estimators': np.arange(50,100,10),
        #'max_depth': range(3,10,1),
        #'num_leaves':range(16, 40, 2),
        #'min_child_samples': np.arange(5, 25),
        #'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95,1.0],
        #'bagging_freq': range(3,8),
        #'feature_fraction': np.arange(0.5,1.0,0.05),
        #'lambda_l1': [0, 0.01, 0.05, 0.1, 0.4, 0.5, 0.6],
        #'lambda_l2': [0, 0.05, 0.1, 0.2, 0.5, 1],
        'cat_smooth': [1,2,3,5, 10, 15, 20, 35]
        }
gbm = lgb.LGBMClassifier(boosting_type='gbdt',objective = 'binary', metric = 'auc', is_unbalance=True,verbose = 1,
                         learning_rate = 0.01,
                         n_estimators = 60,
                         max_depth = 6,
                         num_leaves = 32,
                         min_child_samples = 20,
                         bagging_fraction = 0.9,
                         bagging_freq = 6,
                         feature_fraction =0.65,
                         lambda_l1 = 0,
                         lambda_l2 = 0,
                         cat_smooth = 1
                         )
# 有了gridsearch
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='f1', cv=3,verbose=1, n_jobs=-1)
gsearch.fit(data_train_x,data_train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:",gsearch.best_params_)


#计算模型
model_lgb = lgb.LGBMClassifier(boosting_type='gbdt',objective = 'binary', metric = 'auc', #verbose = 1,is_unbalance=True,
                         learning_rate = 0.01,
                         n_estimators = 60,
                         max_depth = 6,
                         num_leaves = 32,
                         min_child_samples = 20,
                         bagging_fraction = 0.9,
                         bagging_freq = 6,
                         feature_fraction =0.65,
                         lambda_l1 = 0,
                         lambda_l2 = 0,
                         cat_smooth = 1
                               )
model_lgb.fit(data_train_x,data_train_y)
show_accuracy(model_lgb,data_train_x, data_test_x, data_train_y, data_test_y)



####多阶段调参
#一、
params = {    'boosting_type': 'gbdt', 
    'objective': 'binary', 
    'learning_rate': 0.1, 
    'num_leaves': 30, 
    'max_depth': 6,    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    'is_unbalance':True
    }
data_train = lgb.Dataset(data_train_x,data_train_y, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
    early_stopping_rounds=50,verbose_eval=50, show_stdv=True, seed=0)
print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', pd.Series(cv_results['auc-mean']).max())


#二、
model_lgb = lgb.LGBMClassifier(objective='binary',
                              num_leaves=50,
                              learning_rate=0.05, 
                              n_estimators=29, 
                              max_depth=6,
                              metric='auc', 
                              bagging_fraction = 0.8,
                              feature_fraction = 0.8,
                              is_unbalance=True,
                              #scale_pos_weight=40
                              )
params_test1 = {'max_depth': range(3,8,1),
                'num_leaves':range(50, 170, 10)}
gsearch1 = GridSearchCV(estimator=model_lgb, 
                        param_grid=params_test1, 
                        scoring='f1', 
                        cv=5, verbose=1, n_jobs=-1)
gsearch1.fit(data_train_x,data_train_y)
gsearch1.cv_results_['mean_test_score'] , gsearch1.best_params_, gsearch1.best_score_


#三、
params_test3={'min_child_samples': np.arange(5, 25),
              'min_child_weight':np.arange(0, 0.005, 0.001)
              }
model_lgb = lgb.LGBMClassifier(objective='binary',
                               num_leaves=50,
                               learning_rate=0.05, 
                               n_estimators=29, 
                               max_depth=7, 
                               metric='auc', 
                               bagging_fraction = 0.8, 
                               feature_fraction = 0.8,
                               is_unbalance=True,
                              #scale_pos_weight=40
                              )
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='f1', cv=5, verbose=1, n_jobs=-1)
gsearch3.fit(data_train_x,data_train_y)
gsearch3.best_params_, gsearch3.best_score_

#四、
params_test4={
        'feature_fraction': [0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
model_lgb = lgb.LGBMClassifier(objective='binary',num_leaves=50,
                              learning_rate=0.05, n_estimators=29, max_depth=7, 
                              metric='auc', bagging_freq = 5,  min_child_samples=19,
                              is_unbalance=True,
                              #scale_pos_weight=40
                              )
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='f1', cv=5, verbose=1, n_jobs=-1)
gsearch4.fit(data_train_x,data_train_y)
gsearch4.best_params_, gsearch4.best_score_

#五、
params_test6={
        'lambda_l1': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],  
        #'lambda_l1': np.arange(0.2, 0.5, 0.05),  
        'lambda_l2': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
        }
model_lgb = lgb.LGBMClassifier(objective='binary',num_leaves=50,
                              learning_rate=0.05, n_estimators=29, max_depth=7, 
                              metric='auc',  min_child_samples=19, feature_fraction=0.8,bagging_fraction=1,
                              is_unbalance=True,
                              #scale_pos_weight=40
                              )
gsearch6 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='f1', cv=5, verbose=1, n_jobs=-1)
gsearch6.fit(data_train_x,data_train_y)
gsearch6.best_params_, gsearch6.best_score_

    
#六、
params = {'boosting_type': 'gbdt', 'objective': 'binary', 
          'learning_rate': 0.005, 'n_estimators':29,
          'max_depth': 7,'num_leaves': 50,
          'min_child_samples':19,
          'feature_fraction':0.8,'bagging_fraction':1,
          'reg_alpha':0.001,'reg_lambda':0,
          'is_unbalance':True
          #'scale_pos_weight':7
          }
data_train = lgb.Dataset(data_train_x,data_train_y, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='auc',
    early_stopping_rounds=50, verbose_eval=100, show_stdv=True)

print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', cv_results['auc-mean'][-1])


####计算模型
model_lgb = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metric='auc', 
                               learning_rate= 0.001, n_estimators=29,
                               max_depth= 7,num_leaves=50,
                               min_child_samples=19,
                               bagging_fraction = 0.8,feature_fraction = 1,
                               lambda_l1=0,lambda_l2=0.08,
                               is_unbalance=True
                               #scale_pos_weight=40
                               )
model_lgb.fit(data_train_x,data_train_y)
show_accuracy(model_lgb,data_train_x, data_test_x, data_train_y, data_test_y)

