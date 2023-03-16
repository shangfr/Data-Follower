# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:59:38 2023

@author: shangfr
"""
import pickle
from io import BytesIO
import pandas as pd
import streamlit as st


def pickle_model(model):
    '''Pickle the model inside bytes.
    '''
    f = BytesIO()
    pickle.dump(model, f)
    return f

def show_download(cache_data, pre_data):
    '''show download for preprocessing data and trained model.
    '''
    st.sidebar.success('数据、模型和预测结果下载', icon="✅")
    col0, col1, col2 = st.sidebar.columns([1, 1, 1])

    preprocessing_df = pd.DataFrame(cache_data['datasets']['X'])
    sk_model = cache_data['output_pipe']['model']

    col0.download_button(
        label='📝',
        data=preprocessing_df.to_csv(index=False).encode('utf-8'),
        file_name='preprocessing_df.csv',
        mime='text/csv',
        help='download the preprocessing dataframe.'
    )

    col1.download_button(
        label='💠',
        data=pickle_model(sk_model),
        file_name='model.pkl',
        help='download the trained model.'
    )

    col2.download_button(
        label='💎',
        data=pre_data.to_csv(index=False).encode('utf-8'),
        file_name='pre_data.csv',
        mime='text/csv',
        help='download the predict dataframe.'
    )


def show_model(cache_data):
    '''show model.
    '''  
    parm_ml = cache_data['parm_ml']
    parm_model = cache_data['parm_model']
    features = parm_ml['features']
    numerical_cols = features['num_cols']
    categorical_cols = features['cat_cols']
    feature_names = numerical_cols+categorical_cols

    st.success("Model loaded successfully. ", icon="👇")

    if parm_ml["ml_type"] == '无监督':
        content = f'''
        |  Machine Learning   | n_clusters  | features-num  | features-cat  |
        |  ----  | ----  |   ----  |   ----  |
        |  聚类  | {parm_model["n_clusters"]} | {len(numerical_cols)} | {len(categorical_cols)} |
        '''
    elif parm_ml["tgt_type"] == '分类':
        content = f'''
        |  Machine Learning   | target name  | positive  | features-num  | features-cat  |
        |  ----  | ----  |   ----  |   ----  |   ----  |
        | {parm_ml["cls_n"]}{parm_ml["tgt_type"]}  | {parm_ml["target"]} | {parm_ml["positive"]} | {len(numerical_cols)} | {len(categorical_cols)} |
        '''
    elif parm_ml["tgt_type"] == '回归':
        content = f'''
        |  Machine Learning   | target name  | features-num  | features-cat  |
        |  ----  | ----  |   ----  |   ----  |
        | {parm_ml["tgt_type"]}  | {parm_ml["target"]} | {len(numerical_cols)} | {len(categorical_cols)} |
        '''

    if cache_data['datasets'].get('target_names'):
        target_names = cache_data['datasets']['target_names']

        tab1, tab2, tab3 = st.tabs(["model", "features", "target"])

        with tab3:
            st.json(dict(enumerate(target_names)))
    else:
        tab1, tab2 = st.tabs(["model", "features"])

    with tab1:
        st.markdown(content.replace('----', ':----:'))
    with tab2:
        st.json(features) 
        
    return feature_names
        

def predict(output_pipe,df,mtype):
    
    preprocessor = output_pipe['preprocessor']
    model = output_pipe['model']
    df_X = preprocessor.transform(df)
    
    if mtype == '无监督':
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

    feature_names = show_model(cache_data)

    uploaded_file = st.sidebar.file_uploader(
    '上传数据', type=['xlsx', 'csv'])
    if uploaded_file is None:
        st.markdown('---')
        st.warning('请先上传数据集', icon='👈')
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

    st.markdown('---')
    st.success('Prediction Results: ', icon="👇")
    mtype = cache_data['parm_ml']['ml_type']
    output_pipe = cache_data['output_pipe']
    X = df[feature_names]
    df = predict(output_pipe,X,mtype)
    
    st.dataframe(df.style.background_gradient(
        subset=['predict'], cmap='spring'))

    show_download(cache_data, df)