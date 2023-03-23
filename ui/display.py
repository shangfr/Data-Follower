# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:43:11 2023

@author: shangfr
"""
import streamlit as st
from charts import e_bar, e_scatter, heatmap, e_roc, e_pr, e_y_vs


def show_model(cache_data):
    '''show model.
    '''  
    parm_ml = cache_data['parm_ml']
    parm_model = cache_data['parm_model']
    features = parm_ml['features']
    numerical_cols = features['num_cols']
    categorical_cols = features['cat_cols']
    n0 = len(numerical_cols)
    n1 = len(categorical_cols)

    if parm_ml["ml_type"] == '无监督':
        content = f'''
        |  Machine Learning   | n_clusters  | features-num  | features-cat  |
        |  ----  | ----  |   ----  |   ----  |
        |  聚类  | {parm_model["n_clusters"]} | {n0} | {n1} |
        '''
    elif parm_ml["tgt_type"] == '分类':
        content = f'''
        |  Machine Learning   | target name  | positive  | features-num  | features-cat  |
        |  ----  | ----  |   ----  |   ----  |   ----  |
        | {parm_ml["cls_n"]}{parm_ml["tgt_type"]}  | {parm_ml["target"]} | {parm_ml["positive"]} | {n0} | {n1} |
        '''
    elif parm_ml["tgt_type"] == '回归':
        content = f'''
        |  Machine Learning   | target name  | features-num  | features-cat  |
        |  ----  | ----  |   ----  |   ----  |
        | {parm_ml["tgt_type"]}  | {parm_ml["target"]} | {n0} | {n1} |
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
        if n0>0:
            st.caption(f"numerical cols: {'✔️'.join(numerical_cols)}✔️") 
        if n1>0:
            st.caption(f"categorical cols: {'✔️'.join(categorical_cols)}✔️") 
        
    st.markdown('---')
        
        

def result_display(cache_data):
    '''result display.
    '''
    show_model(cache_data)
    st.markdown(cache_data['output_pipe']['report']['score'])
    
    if cache_data['fig_data'].get('cls_report'):
        result = cache_data['fig_data']['cls_report']
        st.markdown('### :orange[classification report]')
        st.text(result)
        st.markdown('---')
    
    
    fig1, fig2 = st.columns([1, 1])
    
    if cache_data['fig_data'].get('feature_importance'):
        result = cache_data['fig_data']['feature_importance']
        with fig1:
            e_bar(result)
    if cache_data['fig_data'].get('cm'):
        result = cache_data['fig_data']['cm']
        with fig2:
            heatmap(result)
    elif cache_data['fig_data'].get('y_vs'):
        result = cache_data['fig_data']['y_vs']
        with fig2:
            e_y_vs(result)

    fig3, fig4 = st.columns([1, 1])
    if cache_data['fig_data'].get('roc'):
        with fig3:
            e_roc(cache_data['fig_data']['roc'])
        with fig4:
            e_pr(cache_data['fig_data']['pr'])

    if cache_data['fig_data'].get('cluster'):
        result = cache_data['fig_data']['cluster']
        e_scatter(result)

                
    
