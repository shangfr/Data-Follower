# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:43:11 2023

@author: shangfr
"""
import streamlit as st
from charts import e_bar, e_scatter, heatmap, e_roc, e_pr, e_y_vs


def result_display(cache_data):
    '''result display.
    '''

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

    st.text(cache_data['output_pipe']['report']['score'])
