# -*- coding: utf-8 -*-
'''
Created on Mon Feb 27 15:49:18 2023

@author: shangfr
'''

from model.supervised import model_cls, model_regr
from model.unsupervised import model_cluster

__version__ = '0.1.1'

sk_models = {'分类': model_cls,
             '回归': model_regr,
             '聚类': model_cluster
             }
__all__ = (
    sk_models
)
