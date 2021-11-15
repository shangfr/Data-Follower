# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:01:33 2021

@author: shangfr
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import roc_curve, auc, classification_report, mean_squared_error, r2_score

from model.ml_plot import plot_roc_curve


def model_cls(model_dict):

    data = model_dict['data']
    target = model_dict['target']
    features = model_dict['features']
    positive = model_dict["positive"]
    
    X = data[features]
    y = data[target]
    y = y == positive
    
    enc = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=np.nan)
    le = LabelEncoder()
    
    X = enc.fit_transform(X)
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('clf', AdaBoostClassifier())])

    pipe.fit(X_train, y_train)

    # Classification metrics
    report = classification_report(y_test, pipe.predict(X_test))

    y_pred = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    auc_value = f'{auc(fpr, tpr):.4f}'

    fig = plot_roc_curve(fpr, tpr, thresholds, auc_value, positive)

    return fig, report


def model_regr(model_dict):

    data = model_dict['data']
    target = model_dict['target']
    features = model_dict['features']
    
    X = data[features]
    y = data[target]
    enc = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=np.nan)
    
    X = enc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('regr', AdaBoostRegressor())])

    pipe.fit(X_train, y_train)
    # Regression metrics
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    result = {"mean_squared_error": mse, "r2_score": r2}
    return result
