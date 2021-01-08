import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

# Basics

string = tf.Variable("this is a string",tf.string)
number = tf.Variable(1,tf.int16)

rank1_tensor = tf.Variable(["Test","sth"],tf.string)
rank2_tensor = tf.Variable([["L1","sth"],["L2","sth"]],tf.string)
tf.rank(rank2_tensor) #Shows the dimension of it
rank2_tensor.shape # Shows [2,2] cf) [nth-dimension,...,1st-dimension]

t1 = tf.ones([1,2,3]) # Indicates [[[1.1.1.],[1.1.1.]]], gotta 1*2*3=6 elements
t2 = tf.reshape(t1,[2,1,3])
t3 = tf.reshape(t2,[3,-1]) # -1 automatically fills the lower dimension according to the number of the elements

with tf.Session() as sess:
    tensor.eval()

# Imprting & Plotting

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain.head() # Same function in R
dftrain.describe() # Simillar with summary in R
dftrain.shape

dftrain.age.hist(bins=20)
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain,y_train],axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('P survive')


