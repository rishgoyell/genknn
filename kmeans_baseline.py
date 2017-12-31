from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import utils
import numpy as np
import pickle
from utils import dataset

M = 15

filename = None
if filename != None:
	with open(filename, 'rb') as f:
		data = pickle.load(f)

for class1 in range(10):
	if filename == None:
		Xtrain,Ytrain = utils.prepare_data('training')
		Xtrain, Ytrain = utils.onevsall(class1,Xtrain,Ytrain)
		Xtest, Ytest = utils.prepare_data(dataset='testing')
		Xtest, Ytest = utils.onevsall(class1,Xtest,Ytest)
	else:
		N = 750
		Xtrain = data.X[:N]
		Ytrain = data.Y[:N]
		Xtest = data.X[N:]
		Ytest = data.Y[N:]

	kmeans = KMeans(n_clusters=M, random_state=0, n_init=5, n_jobs=-2).fit(Xtrain)
	xproto = kmeans.cluster_centers_
	yind = kmeans.labels_
	yproto = np.zeros(M)
	count = np.zeros(M)
	for i in range(yind.shape[0]):
		yproto[yind[i]] = Ytrain[i] + yproto[yind[i]]
		count[yind[i]] = 1 + count[yind[i]]

	yproto = yproto/count
	for k in [1,2,4,8]:
		nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(xproto)
		distances, indices = nbrs.kneighbors(Xtest)
		invdist = np.reciprocal(distances)
		probdist = invdist/np.expand_dims(invdist.sum(axis=1), axis=1)

		ynn = np.zeros([Xtest.shape[0],1])

		for i in range(Xtest.shape[0]):
			for j in range(k):
				ynn[i] = ynn[i]+yproto[indices[i,j]]*probdist[i,j]

		total = ynn.shape[0]
		# print np.sum((ynn[:,0]>0.5)==Ytest)/total, roc_auc_score(Ytest,ynn)
		print np.sum(roc_auc_score(Ytest,ynn)), '& '

