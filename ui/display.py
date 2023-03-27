# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:43:11 2023

@author: shangfr
"""
import streamlit as st
from charts import e_bar, e_scatter, heatmap, e_roc, e_pr, e_y_vs


def show_model(machine_learning):
    '''show model.
    '''
    ml_parm = machine_learning['parm']
    features = ml_parm['features']
    numerical_cols = features['num_cols']
    categorical_cols = features['cat_cols']
    n0 = len(numerical_cols)
    n1 = len(categorical_cols)

    if ml_parm["model_type"] == '聚类':
        content = f'''
        |  Model   | n_clusters  | features-num  | features-cat  |
        |  ----  | ----  |   ----  |   ----  |
        |  聚类  | {ml_parm["n_clusters"]} | {n0} | {n1} |
        '''
    elif ml_parm["model_type"] == '分类':
        content = f'''
        |  Model   | target name  | positive  | features-num  | features-cat  |
        |  ----  | ----  |   ----  |   ----  |   ----  |
        | {ml_parm["cls_n"]}{ml_parm["model_type"]}  | {ml_parm["target"]} | {ml_parm["positive"]} | {n0} | {n1} |
        '''
    elif ml_parm["model_type"] == '回归':
        content = f'''
        |  Model   | target name  | features-num  | features-cat  |
        |  ----  | ----  |   ----  |   ----  |
        | {ml_parm["model_type"]}  | {ml_parm["target"]} | {n0} | {n1} |
        '''

    if ml_parm.get('target_names'):
        target_names = ml_parm['target_names']

        tab1, tab2, tab3 = st.tabs(["model", "features", "target"])

        with tab3:
            st.json(dict(enumerate(target_names)))
    else:
        tab1, tab2 = st.tabs(["model", "features"])

    with tab1:
        st.markdown(content.replace('----', ':----:'))
    with tab2:
        if n0 > 0:
            st.caption(f"numerical cols: {'✔️'.join(numerical_cols)}✔️")
        if n1 > 0:
            st.caption(f"categorical cols: {'✔️'.join(categorical_cols)}✔️")

    st.markdown('---')


def result_display(cache_data):
    '''result display.
    '''
    machine_learning = cache_data['machine_learning']
    output = cache_data['output']
    show_model(machine_learning)
    st.markdown(output['report']['score'])
    if output['report'].get('best_params'):
        st.markdown('### :red[best params]')
        st.json(output['report']['best_params'])
        
    if output['fig_data'].get('cls_report'):
        result = output['fig_data']['cls_report']
        st.markdown('### :orange[classification report]')
        st.text(result)
        st.markdown('---')

    fig1, fig2 = st.columns([1, 1])

    if output['fig_data'].get('feature_importance'):
        result = output['fig_data']['feature_importance']
        with fig1:
            e_bar(result)
    if output['fig_data'].get('cm'):
        result = output['fig_data']['cm']
        with fig2:
            heatmap(result)
    elif output['fig_data'].get('y_vs'):
        result = output['fig_data']['y_vs']
        with fig2:
            e_y_vs(result)

    fig3, fig4 = st.columns([1, 1])
    if output['fig_data'].get('roc'):
        with fig3:
            e_roc(output['fig_data']['roc'])
        with fig4:
            e_pr(output['fig_data']['pr'])

    if output['fig_data'].get('cluster'):
        result = output['fig_data']['cluster']
        e_scatter(result)
