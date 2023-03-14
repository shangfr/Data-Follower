# -*- coding: utf-8 -*-
'''
Created on Mon Feb 27 15:49:18 2023

@author: shangfr
'''

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder


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
        ('imputer', SimpleImputer(strategy='constant')),
        ('scaler', StandardScaler())
    ])

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
    X = data[feature_names]
    datasets['X'] = preprocessor.fit_transform(X)
    datasets['feature_names'] = feature_names

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
