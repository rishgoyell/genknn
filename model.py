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
    MultivariateNormalDiag, Normal, ParamMixture, Bernoulli, 
    Multinomial, PointMass)

import utils
plt.style.use('ggplot')

print("Reading data............")
datareader = utils.readmnist(path='MNIST')
data = []
for i in datareader:
    data.append(i)

Xtrain = np.zeros([60000, 28*28])
Ytrain = np.zeros([60000])

M = 100
N = 60000
D1 = 100
D2 = 28*28                                                                                                                                   
                                                                                                                                                
for i in range(N):                                                                                                                               
    Xtrain[i] = data[i][1].flatten()                                                                                                         
    Ytrain[i] = data[i][0]                                                                                                                   

print("Centering Data........")
Xmean = Xtrain.mean(axis=0)
Xscale = Xtrain.std(axis=0)
np.place(Xscale, Xscale==0, -1)
Xtrain = (Xtrain-Xmean)/Xscale
np.place(Xscale, Xscale==-1, 0)



# model
print("Defining model...........")                                                                                                                                          
beta = Dirichlet(tf.ones(M))                                                                                                                     
mu = Normal(tf.zeros(D1), tf.ones(D1), sample_shape=M)                                                                                           
sigmasq = InverseGamma(tf.ones(D1), tf.ones(D1), sample_shape=M)                                                                                 
z = ParamMixture(beta, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},                                                                              
              MultivariateNormalDiag,                                                                                                         
              sample_shape=N)                                                                                                                 
c = z.cat                                                                                                                                        
wx = Normal(loc=tf.zeros([D2, D1]), scale=tf.ones([D2, D1]))                                                                                     
wy = Normal(loc=tf.zeros([10, D1]), scale=tf.ones([10, D1]))                                                                                     
x = Normal(loc=tf.matmul(z, wx, transpose_b=True), scale=tf.ones([N, D2]))                                                                                         
y = Categorical(logits=tf.matmul(z, wy, transpose_b=True))                                                                                                         

      

# inference
inference = 'EM'

if inference == 'VI':
    qz = Normal(loc=tf.Variable(tf.random_normal([N, D1])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([N, D1]))))
    qmu = Normal(loc=tf.Variable(tf.random_normal([M, D1])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([M, D1]))))
    qwx = Normal(loc=tf.Variable(tf.random_normal([D2, D1])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([D2, D1]))))
    qwy = Normal(loc=tf.Variable(tf.random_normal([10, D1])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([10,D1]))))
    qc = Categorical(logits=tf.Variable(tf.zeros([N,M])))

    # inference = ed.MAP([mu, c, wx, wy, z], data={x: Xtrain, y: Ytrain} )
    inference = ed.KLqp({mu: qmu, c: qc, wx: qwx}, data={x: Xtrain, y: Ytrain})
    #  , z: qz, wy: qwy, wx: qwx
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    inference.run(n_iter=1000, n_print=100, n_samples=20, optimizer=optimizer)


elif inference == 'MAP':
    qz = PointMass(params=tf.Variable(tf.random_normal([N, D1])))
    qmu = PointMass(params=tf.Variable(tf.random_normal([M, D1])))
    qwx = PointMass(params=tf.Variable(tf.random_normal([D2, D1])))
    qwy = PointMass(params=tf.Variable(tf.random_normal([10, D1])))
    qc = PointMass(params=tf.Variable(tf.zeros(N)))
    inference = ed.MAP({mu:qmu,wx:qwx, wy:qwy}, data={x: Xtrain, y: Ytrain})
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    inference.run(n_iter=1000, n_print=100, optimizer=optimizer)

    
if inference == 'EM':
    qz = Normal(loc=tf.Variable(tf.random_normal([N, D1])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([N, D1]))))
    qc = Categorical(logits=tf.Variable(tf.ones([N,M])))
    qmu = PointMass(params=tf.Variable(tf.random_normal([M, D1])))
    qwx = PointMass(params=tf.Variable(tf.random_normal([D2, D1])))
    qwy = PointMass(params=tf.Variable(tf.random_normal([10, D1])))
    qsigmasq = PointMass(params=tf.Variable(tf.ones([M,D1])))
    
    inference_e = ed.KLqp({z:qz,c:qc}, data={x: data.X, y: data.Y, mu:qmu, wx:qwx, wy:qwy, sigmasq:qsigmasq})
    inference_m = ed.MAP({mu:qmu, wx:qwx, wy:qwy, sigmasq:qsigmasq}, data={x: Xtrain, y: Ytrain})

    inference_e.initialize(optimizer = tf.train.AdamOptimizer(learning_rate=1e-3))
    inference_m.initialize()

    init = tf.global_variables_initializer()
    init.run()
    
    for i in range(1000):
        for j in range(10):
            info_dict_m = inference_m.update()
        info_dict_e = inference_e.update()
        inference_m.print_progress(info_dict_m)



sess = ed.get_session()

if inference in ['VI', 'EM']:
    probs = sess.run(qc.probs)
    cluster = np.argmax(probs, axis=1)
    clusterlabels = np.zeros([M, 10])
    for i in range(M):
        temp = Ytrain[np.where(cluster==i)]
        elem, count = np.unique(temp, return_counts=True)
        elem = elem.astype(int)
        for j in range(elem.shape[0]):
            clusterlabels[i,elem[j]] = count[j]
        
zproto = sess.run(qmu.params)
weightx = sess.run(qwx.params)
weighty = sess.run(qwy.params)
xcenters = np.matmul(zproto,weightx.transpose())*Xscale+Xmean
ycenters = np.matmul(zproto, weighty.transpose())

for i in range(xcenters.shape[0]):
    if inference in ['VI', 'EM']:
        print(clusterlabels[i,:].astype(int))
    print(ycenters[i,:])
    # utils.save(xcenters[i,:].reshape((28,28)), 'MNIST/xprotoMAP/'+str(i)+'.png')


# dictionary = np.matmul(zproto,dictionary.transpose())*Xscale+Xmean
# np.place(dictionary, dictionary<0, 0)
# for i in range(dictionary.shape[0]):
#     utils.save(dictionary[i,:].reshape((28,28)), str(i)+'.png')





