# -*- coding: utf-8 -*-
'''
Created on Mon Nov  8 16:01:33 2021

@author: shangfr
'''
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
import numpy as np


def model_cluster(datasets, model_parm):
    '''cluster using sklearn.cluster.
    '''
    X = datasets['X']
    n = model_parm['n_clusters']

    # 步骤二：定义模型，创建管道Pipeline

    #ipca = IncrementalPCA(n_components=2, batch_size=10)
    pca = PCA()
    X_pca = pca.fit_transform(X)

    total_explained_variance = pca.explained_variance_ratio_.cumsum()
    n_over_95 = len(total_explained_variance[total_explained_variance >= .95])
    n_to_reach_95 = X.shape[1] - n_over_95 + 1
    tt = round(total_explained_variance[n_to_reach_95-1]*100,2)
    result = f'🔴 Number of Principal components: {n_to_reach_95}\n🔴 Total Variance Explained: {tt}%'

    kmeans = KMeans(n_clusters=n, random_state=0,
                    n_init='auto').fit(X_pca[:, :n_to_reach_95])

    data = np.insert(X_pca[:, :2].round(2), 2, values=kmeans.labels_, axis=1).tolist()

    fig_data = {'cluster': {'cluster_cnt': n, 'data': data}}
    report = {'score': result}

    return report, fig_data, {'pca': pca, 'kmeans': kmeans,'n_to_reach_95':n_to_reach_95}
