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
import pickle as pk
from sklearn.cluster import KMeans

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, Bernoulli, 
    Multinomial, PointMass, Mixture)

import utils

plt.style.use('ggplot')
                                                                                                                                             
Xtrain,Ytrain = utils.prepare_data('training')                                                                                                                   
Xtrain, Ytrain = utils.onevsone(3,8,Xtrain,Ytrain)

N = Xtrain.shape[0]
M = 10
D1 = 50
D2 = 28*28
K = 2                                                                                                                          

if K==2:
    Ytrain = np.expand_dims(Ytrain, axis=1)
    print(Ytrain.shape)

print("Centering Data........")
# Xmean = Xtrain.mean(axis=0)
# Xscale = Xtrain.std(axis=0)
# np.place(Xscale, Xscale==0, -1)
# Xtrain = (Xtrain-Xmean)/Xscale
# np.place(Xscale, Xscale==-1, 0)

inference = 'EM'
model = 'collapsed'
initialization = 'kmeans'


# model
print("Defining model...........")
if model != 'collapsed':                                                                                                                                          
    beta = Dirichlet(tf.ones(M))                                                                                                                     
    mu = Normal(tf.zeros(D1), tf.ones(D1), sample_shape=M)                                                                                           
    sigmasq = InverseGamma(tf.ones(D1), tf.ones(D1), sample_shape=M)                                                                                 
    z = ParamMixture(beta, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},                                                                              
                  MultivariateNormalDiag,                                                                                                         
                  sample_shape=N)                                                                                                                 
    c = z.cat                                                                                                                                        
    wx = Normal(loc=tf.zeros([D2, D1]), scale=tf.ones([D2, D1]))                                                                                                                                                                         
    x = Normal(loc=tf.matmul(z, wx, transpose_b=True), scale=tf.ones([N, D2]))                                                                                         
    if K == 2:
        wy = Normal(loc=tf.zeros([1, D1]), scale=tf.ones([1, D1]))
        y = Bernoulli(logits=tf.matmul(z, wy, transpose_b=True))
    else:
        wy = Normal(loc=tf.zeros([K, D1]), scale=tf.ones([K, D1]))
        y = Categorical(logits=tf.matmul(z, wy, transpose_b=True))

else:
    beta = Dirichlet(tf.ones(M))
    mu = Normal(tf.zeros(D1), tf.ones(D1), sample_shape=M)
    sigmasq = InverseGamma(tf.ones(D1), tf.ones(D1), sample_shape=M)
    cat = Categorical(probs=beta, sample_shape=N)
    components = [
    MultivariateNormalDiag(mu[k], sigmasq[k], sample_shape=N)
    for k in range(M)]
    z = Mixture(cat=cat, components=components, sample_shape=N)
    wx = Normal(loc=tf.zeros([D2, D1]), scale=tf.ones([D2, D1]))
    x = Normal(loc=tf.matmul(z, wx, transpose_b=True), scale=tf.ones([N, D2]))
    if K == 2:
        wy = Normal(loc=tf.zeros([1, D1]), scale=tf.ones([1, D1]))
        y = Bernoulli(logits=tf.matmul(z, wy, transpose_b=True))
    else:
        wy = Normal(loc=tf.zeros([K, D1]), scale=tf.ones([K, D1]))
        y = Categorical(logits=tf.matmul(z, wy, transpose_b=True))
 

# inference
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

    #initialization
    qz = Normal(loc=tf.Variable(tf.random_normal([N, D1])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([N, D1]))))
    qc = Categorical(logits=tf.Variable(tf.ones([N,M])))

    wxinit = np.random.normal(size=[D2, D1]).astype(np.float32)
    qwx = PointMass(params=tf.Variable(wxinit))

    if K == 2:
        qwy = PointMass(params=tf.Variable(tf.random_normal([1, D1])))
    else:
        qwy = PointMass(params=tf.Variable(tf.random_normal([K, D1])))
        
    qsigmasq = PointMass(params=tf.Variable(tf.ones([M,D1])))

    if initialization == 'random':
        qmu = PointMass(params=tf.Variable(tf.random_normal([M, D1])))
    elif initialization == 'kmeans':
        kmeans = KMeans(n_clusters=M, random_state=0, n_init=5, n_jobs=-2).fit(Xtrain)
        xinit = kmeans.cluster_centers_
        zinit = np.matmul(xinit, np.linalg.pinv(wxinit).transpose()).astype(np.float32)
        qmu = PointMass(params=tf.Variable(zinit))

    qsigmasq = PointMass(params=tf.Variable(tf.ones([M,D1])))
    
    #inference
    inference_e = ed.KLqp({z:qz}, data={x:Xtrain, y:Ytrain, mu:qmu, wx:qwx, wy:qwy, sigmasq:qsigmasq})
    inference_m = ed.MAP({mu:qmu, wx:qwx, wy:qwy, sigmasq:qsigmasq}, data={x: Xtrain, y: Ytrain, z:qz})

    inference_e.initialize(optimizer = tf.train.AdamOptimizer(learning_rate=1e-2))
    inference_m.initialize()

    init = tf.global_variables_initializer()
    init.run()
    
    for i in range(500):
        for j in range(5):
            info_dict_e = inference_e.update()
        info_dict_m = inference_m.update()
        inference_m.print_progress(info_dict_m)



sess = ed.get_session()
        
zproto = sess.run(qmu.params)
weightx = sess.run(qwx.params)
weighty = sess.run(qwy.params)
xcenters = np.matmul(zproto,weightx.transpose())
if K == 2:
    ycenters = 1/(1+np.exp(-np.matmul(zproto,weighty.transpose())))
else:
    uprob = np.exp(np.matmul(zproto,weighty.transpose()))
    ycenters = uprob/np.expand_dims(np.sum(uprob, axis=1), axis=1)
    del uprob
print(xcenters.shape)
print(ycenters.shape)
print(xinit.shape)
xlist = []
ylist = []
if initialization == 'kmeans':
    initlist = []

# for i in range(M):
#     ylist.append(ycenters[i,:])
#     xlist.append(xcenters[i,:].reshape((28,28)))
#     if initialization == 'kmeans':
#         initlist.append(xinit[i,:].reshape((28,28)))

# with open('collapsed'+'_'+str(M)+'_'+str(D1)+'.npz','wb') as outfile:
#     if initialization == 'kmeans':
#         pk.dump([xlist, ylist, initlist], outfile)
#     else:
#         pk.dump([xlist, ylist], outfile)

Xtest, Ytest = utils.prepare_data(dataset='testing')
Xtest, Ytest = utils.onevsone(3,8,Xtest,Ytest)
utils.evaluate(Xtest, Ytest, zproto, weightx, weighty, ycenters,K)
utils.visualize(xcenters, ycenters, xinit)
for k in [1,2,4,8]:
    utils.nnfull(Xtrain, Ytrain, Xtest, Ytest, k)



# if inference in ['VI', 'EM']:
    # probs = sess.run(qc.probs)
    # cluster = np.argmax(probs, axis=1)
    # clusterlabels = np.zeros([M, 10])
    # for i in range(M):
    #     temp = Ytrain[np.where(cluster==i)]
    #     elem, count = np.unique(temp, return_counts=True)
    #     elem = elem.astype(int)
    #     for j in range(elem.shape[0]):
    #         clusterlabels[i,elem[j]] = count[j]