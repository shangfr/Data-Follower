# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:59:38 2023

@author: shangfr
"""
import streamlit as st
import pandas as pd


def model_prediction(cache_data):
    '''Prediction.
    '''
    output_pipe = cache_data['output_pipe']
    preprocessor = output_pipe['preprocessor']
    model = output_pipe['model']
    parm_ml = cache_data['parm_ml']
    st.success("Loaded model!")

    st.json(parm_ml)

    #uploaded_file = 'dataset/iris.xlsx'
    uploaded_file = st.sidebar.file_uploader(
        '上传数据', type=['xlsx', 'csv'], key='predict')
    if uploaded_file is None:
        st.sidebar.warning('请先上传数据集', icon='👆')
    else:
        dtype = uploaded_file.name.split('.')[-1]
        if dtype in ['csv', 'txt']:
            df = pd.read_csv(uploaded_file)
        elif dtype in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)

        st.write("Turning on evaluation mode...")

        st.write("Here's the prediction:")
        features = parm_ml['features']
        numerical_cols = features['num_cols']
        categorical_cols = features['cat_cols']
        feature_names = numerical_cols+categorical_cols
        df_X = preprocessor.transform(df[feature_names])
        if parm_ml['ml_type'] == '无监督':
            pca = model['pca']
            kmeans = model['kmeans']
            n_to_reach_95 = model['n_to_reach_95']
            X_pca = pca.transform(df_X)
            df_pre = kmeans.predict(X_pca[:, :n_to_reach_95])
        else:
            df_pre = model.predict(df_X)
        st.dataframe(df_pre)
