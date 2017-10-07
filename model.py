#%matplotlib inline
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import six
import tensorflow as tf

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, Bernoulli)

import utils
plt.style.use('ggplot')

datareader = utils.readmnist(path='MNIST')
data = []
for i in datareader:
	data.append(i)

Xtrain = np.zeros([60000, 28, 28], dtype=uint8)
Ytrain = np.zeros([60000, 1], dtype=uint8)


M = 1000
N = 60000
D1 = 100
D2 = 28*28

for i in range(N):
	Xtrain[i] = data[i][1]
	Ytrain[i] = data[i][0]

# model
beta = Dirichlet(tf.ones(M))
mu = Normal(tf.zeros(D1), tf.ones(D1), sample_shape=M)
sigmasq = InverseGamma(tf.ones(D1), tf.ones(D1), sample_shape=M)
z = ParamMixture(beta, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                 MultivariateNormalDiag,
                 sample_shape=N)
c = z.cat
wx = Normal(loc=tf.zeros([D1, D2]), scale=tf.ones([D1, D2]))
wy = Normal(loc=tf.zeros([D1, 1]), scale=tf.ones([D1, 1]))
x = Normal(loc=tf.matmul(z, wx), scale=tf.ones([N, D2]))
y = Bernoulli(logits=tf.matmul(z, wy))


# inference
qz = Normal(loc=tf.Variable(tf.random_normal([N, D1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([N, D1]))))
qmu = Normal(loc=tf.Variable(tf.random_normal([M, D1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([M, D1]))))
qwx = Normal(loc=tf.Variable(tf.random_normal([D1, D2])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D1, D2]))))
qwy = Normal(loc=tf.Variable(tf.random_normal([D1, 1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D1, 1]))))
qc = Categorical(logits=tf.Variable(tf.zeros([N,M])))

inference = ed.KLqp({mu: qmu, c: qc}, data={x: Xtrain, y: Ytrain})
#  , z: qz, wy: qwy, wx: qwx
inference.run(n_iter=10000, n_print=100, n_samples=20)