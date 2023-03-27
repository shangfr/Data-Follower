# -*- coding: utf-8 -*-
'''
Created on Mon Oct 25 11:35:11 2021

@author: shangfr
'''
import streamlit as st
from model import sk_models
from utils import pickle_model, pickle_cache


def reset_step():
    '''reset_step.
    '''
    st.session_state['ml_step'] = 2


@st.cache_resource
def train(datasets, model_parm):
    '''Training base on ML type.
    '''
    model_type = model_parm['model_type']
    model_run = sk_models.get(model_type)
    report, fig, sk_model = model_run(datasets, model_parm)

    return report, fig, sk_model


def model_setup(model_dict):
    '''select score criterion.
    '''
    col1, col2, col3 = st.columns(3)

    model_type = model_dict['model_type']
    if model_type == '分类':
        score_criterion = col1.selectbox(
            '评分准则', ['accuracy', 'precision', 'recall'], key='score_criterion_c', on_change=reset_step)
        model_dict['score_criterion'] = score_criterion
    elif model_type == '回归':
        score_criterion = col1.selectbox(
            '评分准则', ['mean_squared_error', 'mean_pinball_loss'], key='score_criterion_r', on_change=reset_step)
        model_dict['score_criterion'] = score_criterion
    elif model_type == '聚类':
        max_n = model_dict['max_n']
        model_dict['n_clusters'] = col1.number_input(
            '聚类数目', 2, min(max_n, 10), key='n_clusters', on_change=reset_step)

    return model_dict


def data_modeling(cache_data):
    '''Training and Evaluation.
    '''
    st.info('4. 模型训练(Training)', icon='👇')
    machine_learning = cache_data['machine_learning']
    parm_model = machine_learning['parm']
    parm_model = model_setup(parm_model)

    if st.button('🔧 训练'):
        datasets = machine_learning['datasets']

        report, fig, sk_model = train(datasets, parm_model)

        machine_learning['model_pipe']['model'] = sk_model
        output = cache_data['output']
        output['report'] = report
        output['fig_data'] = fig
        st.session_state['ml_step'] = 3

    if st.session_state['ml_step'] == 2:
        st.warning('请点击🔧进行模型训练', icon='⚠️')
        st.stop()

    col0, col1 = st.sidebar.columns([1, 5])
    col1.success('已完成模型训练')
    sk_model = machine_learning['model_pipe']['model']
    col0.download_button(
        label='💠',
        data=pickle_model(sk_model),
        file_name='model.pkl',
        help='download the trained model.'
    )

    # 模型保存
    title = st.sidebar.text_input(
        '👇 Enter a name to save the model', '', help='以pickle的格式保存所有缓存数据')
    if title:
        pickle_cache(f'tmp/{title}.pkl')
        st.sidebar.info('Model saved successfully.')
