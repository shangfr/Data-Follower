# -*- coding: utf-8 -*-
'''
Created on Mon Feb 27 15:49:18 2023

@author: shangfr
'''
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.covariance import empirical_covariance

def x2cor(xarray):  
    # #############################################################################
    # Learn a graphical structure from the correlations
    xarray = StandardScaler().fit_transform(xarray)
    emp_cov = empirical_covariance(xarray)

    d = 1 / np.sqrt(np.diag(emp_cov))
    emp_cov *= d
    emp_cov *= d[:, np.newaxis]
    emp_cor = np.around(emp_cov, decimals=3)

    zero = (np.abs(np.triu(emp_cor, k=1)) < 0.1)        
    emp_cor[zero] = 0
    
    # 寻找非全零列
    non_zero = np.where(emp_cor.any(axis=0))[0]
    if non_zero:
        output = {'cor':emp_cor[:,non_zero][non_zero,:].tolist(),'non_zero':non_zero.tolist()}
    else:
        output = {'cor':emp_cor.tolist(),'non_zero':list(range(len(emp_cor)))}
    return output

def transformer(cache_data):
    '''transformer data using sklearn.preprocessing.
    '''
    data = cache_data['origin']['data']
    data.dropna(axis=1, how='all', inplace=True)
    data.drop_duplicates(subset=None, keep='first', inplace=True)
    
    parm_ml = cache_data['parm_ml']
    features = parm_ml['features']
    numerical_cols = features['num_cols']
    categorical_cols = features['cat_cols']

    # 步骤一：确定预处理步骤

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant'))
    ])
    
    #
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])


    datasets = {}
    feature_names = numerical_cols+categorical_cols
    X = preprocessor.fit_transform(data[feature_names])
    datasets['X'] = X
    datasets['feature_names'] = feature_names
    cor_dict = x2cor(X)
    cor_dict['cls_names'] = [feature_names[i] for i in cor_dict['non_zero']]
        
    datasets['cor_dict'] = cor_dict
    
    ml_type = parm_ml['ml_type']
    if ml_type == '有监督':
        target = parm_ml['target']
        tgt_type = parm_ml['tgt_type']
        if tgt_type == '分类':
            positive = parm_ml['positive']
            y = data[target]
            le = LabelEncoder()
            #y = y == positive
            
            y = le.fit_transform(y)
            
            datasets['target_names'] = list(map(str,le.classes_))
            datasets['pos_id'] = le.transform([positive])[0]
        elif tgt_type == '回归':
            y = data[target].values

        datasets['y'] = y
        
        
        
    return datasets,preprocessor
