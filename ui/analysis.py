# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:57:37 2023

@author: shangfr
"""
import pandas as pd
import streamlit as st
from charts import heatmap


@st.cache_data(show_spinner=False)
def tgt2group(data, target, model_type):
    '''data group by target.
    '''
    if model_type == '回归':
        data[target] = pd.qcut(data[target], 4, labels=[
                               "Q1", "Q2", "Q3", "Q4"])

    return data.groupby(target)


def view_features(dta, parm_ml, option_f):
    '''data analysis.
    '''
    data = dta.copy()
    features = parm_ml['features']

    if option_f in features['num_cols']:
        if data[option_f].nunique() > 10:
            data[option_f] = pd.qcut(data[option_f], 4, labels=[
                                     "Q1", "Q2", "Q3", "Q4"])
    model_type = parm_ml['model_type']
    if model_type != '聚类':
        target = parm_ml['target']
        dgroup = tgt2group(data, target, model_type)
        dfb = dgroup[option_f].value_counts().unstack().T
        dfb.index.name = None
    else:
        dfb = data[option_f].value_counts()

    st.bar_chart(dfb)


def data_analysis(cache_data):
    '''data analysis.
    '''
    data = cache_data['origin']['data']
    machine_learning = cache_data['machine_learning']
    ml_parm = machine_learning['parm']
    datasets = machine_learning['datasets']
    feature_names = ml_parm['feature_names']
    cor_matrix = datasets['cor_matrix']
    col0, col1 = st.columns(2)
    option_t = col0.selectbox('Method', ["Distribution", "Correlation"])
    if option_t == 'Distribution':
        option_f = col1.selectbox('Feature', feature_names)
        view_features(data, ml_parm, option_f)
    elif option_t == 'Correlation':

        result = {'data': cor_matrix.tolist(), 'classes': feature_names,
                  'title': 'Correlation Matrix'}

        heatmap(result)
