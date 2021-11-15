# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:01:33 2021

@author: shangfr
"""

import plotly.express as px


def plot_roc_curve(fpr, tpr, thresholds, auc, positive):

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc})---(True Positive:{positive})',
        labels=dict(x='False Positive Rate', y='True Positive Rate')
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    return fig


def plot_scatter(components, x, y):
    fig = px.scatter(components, x, y)
    return fig
