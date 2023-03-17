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
        "grid": {"left": "3%", "right": "4%", "bottom": "0%", "containLabel": True},
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
        height="300px",
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
            'width':'90%',
            'left': 120
        },
        'xAxis': {},
        'yAxis': {},
        'series': {
            'type': 'scatter',
            'encode': {'tooltip': [0, 1]},
            'symbolSize': 10,
            'itemStyle': {
                'borderColor': '#555'
            },
            'data': data
        }
    }
    st_echarts(options=option, height="400px")


def heatmap(result):
    '''echarts heatmap.
    '''
    xy_label = result['classes']
    c_m = result['data']
    data_min = min(min(c_m))
    data_max = max(max(c_m))
    data = []
    for i in range(len(c_m)):
        for j in range(len(c_m[i])):
            data.append([i, j, c_m[i][j]])

    text_style = {
        'color': '#000',
        'fontSize': 12,
    }
    title = result['title']
    xname = ''
    yname = ''
    show = False
    if title == 'Confusion Matrix':
        xname = 'Predicted label'
        yname = 'True label'
        show = True
    option = {
        'title': {
            'text': title
        },
        "tooltip": {"position": "top"},
        "grid": {"left": "0%", "right": "0%", "bottom": "0%", "containLabel": True},
        "xAxis": {'name': xname, 'nameLocation': "middle", 'nameTextStyle': text_style, "type": "category", "data": xy_label, "splitArea": {"show": True}},
        "yAxis": {'name': yname, 'nameLocation': "middle", 'nameTextStyle': text_style, "type": "category", "data": xy_label, "splitArea": {"show": True}},
        "visualMap": {
            "min": data_min,
            "max": data_max,
            "orient": "vertical",
            "left": "right",
            "top": "center",
        },
        "series": [
            {
                "name": title,
                "type": "heatmap",
                "data": data,
                "label": {"show": show},
                "emphasis": {
                    "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                },
            }
        ],
    }
    st_echarts(option, height="300px")


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
        "grid": {"left": "3%", "right": "4%", "bottom": "0%", "containLabel": True},
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
                'itemStyle': {
                    'color': 'rgb(131, 255, 70)'
                },
                'areaStyle': {},
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
        "grid": {"left": "3%", "right": "4%", "bottom": "0%", "containLabel": True},
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
                'itemStyle': {
                    'color': 'rgb(255, 70, 131)'
                },
                'areaStyle': {},
                'smooth': True,
                'symbolSize': 1,
                'data': data
            }
        ]
    }
    st_echarts(options=options, height="300px")


def e_y_vs(result):
    y_true = result['y_true']
    y_pred = result['y_pred']
    x = list(range(len(y_true)))
    options = {
        "title": {"text": "Target"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["true", "pred"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "0%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": x,
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "name": "true",
                "type": "line",
                "data": y_true,
            },
            {
                "name": "pred",
                "type": "line",
                "data": y_pred,
            },
        ],
    }
    st_echarts(options=options, height="300px")