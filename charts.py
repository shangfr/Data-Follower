# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:58:31 2023

@author: shangfr
"""

from streamlit_echarts import st_echarts


def e_bar(result):
    '''echarts bar.
    '''
    title = 'Feature importance'
    options = {
        'title': {
            'text': title
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            }
        },
        "xAxis": {
            "type": "category",
            "data": result['names'],
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "data": result['importance'],
                "type": "bar",
            }
        ],
    }
    st_echarts(
        options=options,
        height="400px",
    )


def e_scatter(result):
    '''echarts scatter.
    '''
    data = result['data']
    CLUSTER_COUNT = result['cluster_cnt']

    COLOR_ALL = [
        '#37A2DA',
        '#e06343',
        '#37a354',
        '#b55dba',
        '#b5bd48',
        '#8378EA',
        '#96BFFF',
        '#ef5b9c',
        '#f58220',
        '#7fb80e'
    ]

    pieces = []

    for i in range(CLUSTER_COUNT):
        pieces.append({
            'value': i,
            'label': f'cluster {i}',
            'color': COLOR_ALL[i]
        })

    title = 'Kmeans'
    option = {
        'title': {
            'text': title
        },
        'tooltip': {
            'position': 'top'
        },
        'visualMap': {
            'type': 'piecewise',
            'top': 'middle',
            'min': 0,
            'max': CLUSTER_COUNT,
            'left': 10,
            'splitNumber': CLUSTER_COUNT,
            'dimension': 2,
            'pieces': pieces
        },
        'grid': {
            'left': 120
        },
        'xAxis': {},
        'yAxis': {},
        'series': {
            'type': 'scatter',
            'encode': {'tooltip': [0, 1]},
            'symbolSize': 15,
            'itemStyle': {
                'borderColor': '#555'
            },
            'data': data
        }
    }
    st_echarts(options=option, height="500px")


def heatmap(result):
    '''echarts heatmap.
    '''
    xy_label = result['classes']
    c_m = result['data']
    data = []
    for i in range(len(c_m)):
        for j in range(len(c_m[i])):
            data.append([i, j, c_m[i][j]])

    text_style = {
        'color': '#000',
        'fontSize': 16,
    }
    title = 'Confusion matrix'
    option = {
        'title': {
            'text': title
        },
        "tooltip": {"position": "top"},
        "grid": {"height": "50%", "top": "10%"},
        "xAxis": {'name': 'Predicted label', 'nameLocation': "middle", 'nameTextStyle': text_style, "type": "category", "data": xy_label, "splitArea": {"show": True}},
        "yAxis": {'name': 'True label', 'nameLocation': "middle", 'nameTextStyle': text_style, "type": "category", "data": xy_label, "splitArea": {"show": True}},
        "visualMap": {
            "min": 0,
            "max": max(max(data)),
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": "15%",
        },
        "series": [
            {
                "name": "混淆矩阵",
                "type": "heatmap",
                "data": data,
                "label": {"show": True},
                "emphasis": {
                    "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                },
            }
        ],
    }
    st_echarts(option, height="500px")


def e_roc(result):
    '''echarts roc line.
    '''
    fpr = result['fpr']
    tpr = result['tpr']
    auc = result['AUC']
    positive = result['positive']

    title = f'ROC Curve (AUC={auc})'
    subtext = f'True Positive:{positive}'
    data = []
    for i in range(len(fpr)):
        data.append([fpr[i], tpr[i]])

    options = {
        'title': {
            'text': title,
            'subtext': subtext,
        },

        'xAxis': {
            'min': 0,
            'max': 1,
            'type': 'value'
        },
        'yAxis': {
            'min': 0,
            'max': 1,
            'type': 'value'
        },
        'series': [
            {
                'type': 'line',
                'smooth': True,
                'symbolSize': 1,
                'data': data
            }
        ]
    }
    st_echarts(options=options, height="300px")


def e_pr(result):
    '''echarts pr line.
    '''
    recall = result['recall']
    prec = result['prec']
    ap = result['AP']

    data = []
    for i in range(len(prec)):
        data.append([recall[i], prec[i]])

    title = f'PR Curve (AP={ap})'
    options = {
        'title': {
            'text': title
        },

        'xAxis': {
            'min': 0,
            'max': 1,
            'type': 'value'
        },
        'yAxis': {
            'min': 0,
            'max': 1,
            'type': 'value'
        },
        'series': [
            {
                'type': 'line',
                'smooth': True,
                'symbolSize': 1,
                'data': data
            }
        ]
    }
    st_echarts(options=options, height="300px")
