# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:57:37 2023

@author: shangfr
"""
import pandas as pd
import streamlit as st
from charts import heatmap

@st.cache_data(show_spinner=False)
def tgt2group(data,target,tgt_type):
    '''data group by target.
    '''
    if tgt_type == '回归':
        data[target] = pd.qcut(data[target],4,labels=["Q1","Q2","Q3","Q4"])
        
    return data.groupby(target)


def view_features(dta,parm_ml,option_f):
    '''data analysis.
    '''    
    data = dta.copy()
    features = parm_ml['features']

    if option_f in features['num_cols']:
        if data[option_f].nunique() > 10:
            data[option_f] = pd.qcut(data[option_f],4,labels=["Q1","Q2","Q3","Q4"])

    if parm_ml['ml_type'] == '有监督':
        target = parm_ml['target']
        tgt_type = parm_ml['tgt_type']
        dgroup = tgt2group(data,target,tgt_type)
        dfb = dgroup[option_f].value_counts().unstack().T
        dfb.index.name=None
    else:
        dfb = data[option_f].value_counts()

    st.bar_chart(dfb)
        
def data_analysis(cache_data):
    '''data analysis.
    '''  
    data = cache_data['origin']['data']
    parm_ml = cache_data['parm_ml']
    datasets = cache_data['datasets']
    feature_names = parm_ml['feature_names']
    
    
    col0,col1 = st.columns(2)
    option_t = col0.selectbox('Method',["Distribution", "Correlation"])
    if option_t == 'Distribution':
        option_f = col1.selectbox('Feature',feature_names)
        view_features(data,parm_ml,option_f)
    elif option_t == 'Correlation':
        
        result = {'data': pd.DataFrame(datasets['X']).corr().round(2).values.tolist(
        ), 'classes': feature_names, 'title': 'Correlation Matrix'}
        
        heatmap(result)