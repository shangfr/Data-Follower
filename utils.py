# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:17:59 2023

@author: shangfr
"""
import os
import pickle
import pandas as pd
from io import BytesIO
import streamlit as st


def describe(df):
    '''describe.
    '''
    dts = df.dtypes
    for col in dts[df.dtypes == 'object'].index:
        df[col] = df[col].astype('category')
    dts = df.dtypes
    s0 = df.nunique(axis=0, dropna=True)
    s1 = df.notnull().sum()
    s2 = df.isnull().sum()/df.shape[0]

    df_dt = pd.DataFrame(
        {'dtypes': dts.values.astype(str),
         'var_count': s0,
         'notnull': s1,
         'na_ratio': s2.round(2),
         'effective': [True]*len(dts)
         }, index=dts.index)

    # 常量值、单类别比例>95%、缺失率>95%的变量设为无效

    filter0 = s0 == 1
    filter10 = s0/s1 >= 0.95
    filter11 = df_dt['dtypes'] == 'category'
    filter2 = s2 >= 0.95

    filters = filter0 | (filter10 & filter11) | filter2

    df_dt.loc[filters, 'effective'] = False

    df_dt.index.name = 'variable'
    df_dt.reset_index(inplace=True)

    output = {'data': df, 'dtype_table': df_dt}
    return output


@st.cache_resource
def pickle_load(files_opt):
    with open('tmp/' + files_opt, 'rb') as fr:
        cache_data = pickle.load(fr)
    return cache_data


def load_pickle():
    '''load pickle.
    '''
    files = os.listdir('tmp')
    files_opt = st.sidebar.selectbox('模型选择:', files)
    if not files_opt:
        st.warning('No trained model found!', icon="⚠️")
        st.stop()
    cache_data = pickle_load(files_opt)
    return cache_data


def pickle_cache(file_name='dict_file.pkl'):
    '''save cache_data.
    '''
    with open(file_name, 'wb') as f_save:
        pickle.dump(st.session_state['cache_data'], f_save)


def pickle_model(model):
    '''Pickle the model inside bytes.
    '''
    f = BytesIO()
    pickle.dump(model, f)
    return f


def show_download(cache_data):
    '''show download for preprocessing data and trained model.
    '''
    st.sidebar.success('数据、模型和预测结果下载', icon="✅")
    col0, col1, col2 = st.sidebar.columns([1, 1, 1])

    preprocessing_df = pd.DataFrame(cache_data['datasets']['X'])

    col0.download_button(
        label='📝',
        data=preprocessing_df.to_csv(index=False).encode('utf-8'),
        file_name='preprocessing_df.csv',
        mime='text/csv',
        help='download the preprocessing dataframe.'
    )

    if cache_data['output_pipe'].get('model'):
        sk_model = cache_data['output_pipe']['model']
        col1.download_button(
            label='💠',
            data=pickle_model(sk_model),
            file_name='model.pkl',
            help='download the trained model.'
        )

    if cache_data.get('predict'):
        col2.download_button(
            label='💎',
            data=cache_data['predict'].to_csv(index=False).encode('utf-8'),
            file_name='pre_data.csv',
            mime='text/csv',
            help='download the predict dataframe.'
        )
