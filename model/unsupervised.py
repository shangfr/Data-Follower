# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:01:33 2021

@author: shangfr
"""


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from model.ml_plot import plot_scatter


def model_cluster(model_dict):

    data = model_dict['data']

    pipe = Pipeline([('encoder', OrdinalEncoder()),
                     ('scaler', StandardScaler()),
                     ('pca', PCA(n_components=2))])

    components = pipe.fit_transform(data)

    fig = plot_scatter(components, x=0, y=1)

    return fig
