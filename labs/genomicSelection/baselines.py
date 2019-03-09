'''
File: baselines.py
Project: deepgs
File Created: Tuesday, 5th February 2019 11:33:06 am
Author: Romain GAUTRON (r.gautron@cgiar.org)
-----
Last Modified: Tuesday, 5th February 2019 11:33:11 am
Modified By: Romain GAUTRON (r.gautron@cgiar.org>)
-----
Copyright 2019 Romain GAUTRON, CIAT
'''
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics import mean_absolute_error
i = 123

def main():
    X = pd.read_csv('X.csv',header=0).as_matrix()
    y = pd.read_csv('Y.csv',header=0)['length'].as_matrix()
    modelNames=['randomForest','SVR']
    modelDic={'SVR':SVR(verbose=1),'randomForest':RandomForestRegressor(verbose=1,n_jobs=-1)}
    paramDic = {
        'SVR':{'C': [1, 10, 100],'gamma': [.01, .1]},\
        'randomForest':{'n_estimators': [200, 500],'max_features': ['auto', 'sqrt', 'log2']}
        }
    nestedScoreDic={modelName:0 for modelName in modelNames }
    innerCv = KFold(n_splits=5, shuffle=True, random_state=i)
    outerCv = KFold(n_splits=5, shuffle=True, random_state=i)
    count=0
    for trainOuterIndex, testOuterIndex in outerCv.split(X):
        print('OuterCV step #{}'.format(count))
        count+=1
        XtrainOuter, XtestOuter = X[trainOuterIndex], X[testOuterIndex]
        yTrainOuter, yTestOuter = y[trainOuterIndex], y[testOuterIndex]
        for modelName in modelNames:
            reg  = GridSearchCV(estimator=modelDic[modelName], param_grid=paramDic[modelName], cv=innerCv, scoring='neg_mean_absolute_error', verbose=3,n_jobs=-1)
            reg.fit(XtrainOuter, yTrainOuter)
            nestedScoreDic[modelName]+= mean_absolute_error(yTestOuter,reg.predict(XtestOuter))
    nestedScoreDic = {modelName:sumMAE / outerCv.get_n_splits(X) for modelName, sumMAE in nestedScoreDic.items()}
    pprint(nestedScoreDic)

if __name__ == '__main__':
    main()