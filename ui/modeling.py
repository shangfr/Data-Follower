# -*- coding: utf-8 -*-
'''
Created on Mon Oct 25 11:35:11 2021

@author: shangfr
'''
import json
import streamlit as st
from model import model_cls, model_regr, model_cluster
from utils import pickle_model, pickle_cache


@st.cache_resource
def train(datasets, model_parm):
    '''Training base on ML type.
    '''
    if model_parm['ml_type'] == '无监督':
        model_run = model_cluster
    else:
        if model_parm['tgt_type'] == '分类':
            model_run = model_cls
        else:
            model_run = model_regr

    report, fig, sk_model = model_run(datasets, model_parm)

    return report, fig, sk_model


def model_setup(model_dict):
    '''select score criterion.
    '''
    col1, col2, col3 = st.columns(3)

    ml_type = model_dict['ml_type']
    if ml_type == '有监督':
        tgt_type = model_dict['tgt_type']
        if tgt_type == '分类':
            if model_dict.get('score_criterion') and 'score_criterion_c' not in st.session_state:
                st.session_state.score_criterion_c = model_dict['score_criterion']
            score_criterion = col1.selectbox(
                '评分准则', ['accuracy', 'precision', 'recall'], key='score_criterion_c')
        else:
            if model_dict.get('score_criterion') and 'score_criterion_r' not in st.session_state:
                st.session_state.score_criterion_r = model_dict['score_criterion']
            score_criterion = col1.selectbox(
                '评分准则', ['mean_squared_error', 'mean_pinball_loss'], key='score_criterion_r')
        model_dict['score_criterion'] = score_criterion
    else:
        if model_dict.get('n_clusters') and 'n_clusters' not in st.session_state:
            st.session_state.n_clusters = model_dict['n_clusters']

        max_n = model_dict['max_n']
        model_dict['n_clusters'] = col1.number_input(
            '聚类数目', 2, min(max_n, 10), key='n_clusters')

    return model_dict


def data_modeling(cache_data):
    '''Training and Evaluation.
    '''
    st.info('4. 模型训练(Training)', icon='👇')
    parm_model = cache_data['parm_model']
    parm_model['ml_type'] = cache_data['parm_ml']['ml_type']
    parm_model['tgt_type'] = cache_data['parm_ml']['tgt_type']
    parm_model['max_n'] = cache_data['parm_ml']['max_n']

    hash_value = hash(json.dumps(parm_model))

    cache_data['parm_model'] = model_setup(parm_model)

    if hash_value != hash(json.dumps(cache_data['parm_model'])):
        st.session_state['ml_step'] = 2
    if st.button('🔧 训练'):
        datasets = cache_data['datasets']

        report, fig, sk_model = train(datasets, parm_model)

        cache_data['output_pipe']['model'] = sk_model
        cache_data['output_pipe']['report'] = report
        cache_data['fig_data'] = fig
        st.session_state['ml_step'] = 3

    if st.session_state['ml_step'] == 2:
        st.warning('请点击🔧进行模型训练', icon='⚠️')
        st.stop()
    
    col0, col1 = st.sidebar.columns([1, 5])
    col1.success('已完成模型训练')
    sk_model = cache_data['output_pipe']['model']
    col0.download_button(
        label='💠',
        data=pickle_model(sk_model),
        file_name='model.pkl',
        help='download the trained model.'
    )

    # 模型保存
    title = st.sidebar.text_input('👇 Enter a name to save the model', '', help = '以pickle的格式保存所有缓存数据')
    if title:
        pickle_cache(f'tmp/{title}.pkl')
        st.sidebar.info('Model saved successfully.')
                