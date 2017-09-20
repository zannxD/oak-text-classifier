# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:54:41 2017

@author: deeplearning
"""

import mxnet as mx

prefix = 'cnn'
iteration = 49
model = mx.model.load_checkpoint(prefix,iteration)

#model.fit()

#mod.predict("Love this movie");

