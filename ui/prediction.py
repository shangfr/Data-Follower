# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:59:38 2023

@author: shangfr
"""
import pandas as pd
import streamlit as st


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


def predict(output_pipe, df, mtype):

    preprocessor = output_pipe['preprocessor']
    model = output_pipe['model']
    df_X = preprocessor.transform(df)

    if mtype == '聚类':
        pca = model['pca']
        kmeans = model['kmeans']
        n_to_reach_95 = model['n_to_reach_95']
        X_pca = pca.transform(df_X)
        df_pre = kmeans.predict(X_pca[:, :n_to_reach_95])
    else:
        df_pre = model.predict(df_X)

    df.insert(0, "predict", df_pre)

    return df


def model_prediction(cache_data):
    '''Prediction.
    '''
    machine_learning = cache_data['machine_learning']
    parm_ml = machine_learning['parm']
    features = parm_ml['features']
    numerical_cols = features['num_cols']
    categorical_cols = features['cat_cols']
    feature_names = numerical_cols+categorical_cols

    uploaded_file = st.file_uploader(
        '上传数据', type=['xlsx', 'csv'])
    if uploaded_file is None:
        st.warning(f"请先上传数据集，需要特征名称： {'✔️'.join(feature_names)}✔️", icon='👆')
        st.stop()

    dtype = uploaded_file.name.split('.')[-1]
    if dtype in ['csv', 'txt']:
        df = pd.read_csv(uploaded_file)
    elif dtype in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)
    cols = df.columns.tolist()

    check_col = [f for f in feature_names if f in cols]
    leak_col = [f for f in feature_names if f not in cols]

    if len(check_col) == 0:
        st.markdown('---')
        st.error('features are not found in the upload data.', icon='🚨')
        st.json(leak_col)
        st.stop()

    elif len(leak_col) != 0:
        st.markdown('---')
        st.warning(f'Missing features {str(leak_col)} ', icon='⚠️')
        df[leak_col] = 0

    mtype = parm_ml['model_type']
    output_pipe = machine_learning['model_pipe']
    X = df[feature_names]
    df = predict(output_pipe, X, mtype)

    st.write("### 👇 Prediction Results", df.style.background_gradient(
        subset=['predict'], cmap='spring'))

    col0, col1 = st.sidebar.columns([1, 5])
    col1.success('已完成预测')
    col0.download_button(
        label='💎',
        data=convert_df(df),
        file_name='pre_data.csv',
        mime='text/csv',
        help='download the predict dataframe.'
    )
