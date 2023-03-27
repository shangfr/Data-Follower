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


def x2cor(xarray, r=0.1):
    # #############################################################################
    # Learn a graphical structure from the correlations
    xarray = StandardScaler().fit_transform(xarray)
    emp_cov = empirical_covariance(xarray)

    d = 1 / np.sqrt(np.diag(emp_cov))
    emp_cov *= d
    emp_cov *= d[:, np.newaxis]
    emp_cor = np.around(emp_cov, decimals=3)

    zero = (np.abs(np.triu(emp_cor, k=1)) < r)
    emp_cor[zero] = 0

    return emp_cor


def transformer(data, parm_ml):
    '''transformer data using sklearn.preprocessing.
    '''

    data.dropna(axis=1, how='all', inplace=True)
    data.drop_duplicates(subset=None, keep='first', inplace=True)

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
    datasets['cor_matrix'] = x2cor(X)

    parm_ml['feature_names'] = feature_names
    model_type = parm_ml['model_type']
    target = parm_ml['target']

    if model_type == '分类':
        positive = parm_ml['positive']
        y = data[target]
        le = LabelEncoder()
        #y = y == positive

        y = le.fit_transform(y)

        parm_ml['target_names'] = list(map(str, le.classes_))
        parm_ml['pos_id'] = le.transform([positive])[0]
    elif model_type == '回归':
        y = data[target].values

    if target:
        datasets['y'] = y

    return datasets, preprocessor
