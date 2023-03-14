# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:01:33 2021

@author: shangfr
"""

from ui.exploration import data_exploration
from ui.modeling import data_modeling
from ui.prediction import model_prediction

__version__ = '0.1.1'

__all__ = (
    data_exploration,
    data_modeling,
    model_prediction
)
