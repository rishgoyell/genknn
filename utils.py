import os
import struct
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def readmnist(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    # else:
    #     raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)


def prepare_data(dataset='training'):
    print("Reading data............")
    datareader = readmnist(path='MNIST', dataset=dataset)
    data = []
    for i in datareader:
        data.append(i)

    N = len(data)
    X = np.zeros([N, 28*28])
    Y = np.zeros([N])
    for i in range(N):                                                                                                                               
        X[i] = data[i][1].flatten()                                                                                                         
        Y[i] = data[i][0]
    return X, Y


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def save(image, name):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.savefig(name)


def onevsone(a,b,X,Y):
    A = np.asarray(Y==a)
    B = np.asarray(Y==b)
    C = np.where((A+B)==True)
    temp = Y[C]
    np.place(temp, temp==a, 0)
    np.place(temp, temp==b, 1)
    return X[C][:200,:], temp[:200]

def onevsall(a,X,Y):
    A = Y[np.where(Y==a)]
    B = Y[np.where(Y!=a)]
    num = A[0].shape[0]
    np.random.shuffle(B[0])
    C = np.concatenate((A[0], B[0][:2*num]), axis=0)
    np.random.shuffle(C)
    temp = Y[C]
    np.place(temp, temp==a, 0)
    np.place(temp, temp!=a, 1)
    return X[C][:100,:], temp[:100]


def evaluate(Xtest, Ytest, zproto, weightx, weighty, ycenters, numclasses):

    invWx = np.linalg.pinv(weightx)
    ztest = np.matmul(Xtest, invWx.transpose())

    if numclasses == 2:
        ymat = 1/(1+np.exp(-np.matmul(ztest,weighty.transpose())))
    else:
        uprob = np.exp(np.matmul(ztest,weighty.transpose()))
        ymat = uprob/np.expand_dims(np.sum(uprob, axis=1), axis=1)
        del uprob
    
    for k in [1,2,4,8]:
        print(k)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(zproto)
        distances, indices = nbrs.kneighbors(ztest)
        invdist = np.reciprocal(distances)
        probdist = invdist/np.expand_dims(invdist.sum(axis=1), axis=1)
        
        if numclasses==2:
            ynn = np.zeros([Xtest.shape[0],1])
        else:
            ynn = np.zeros([Xtest.shape[0],numclasses])
            countnn = 0
            countmat = 0
        
        for i in range(Xtest.shape[0]):
            for j in range(k):
                ynn[i] = ynn[i]+ycenters[indices[i,j]]*probdist[i,j]
            if numclasses > 2:
                if np.argmax(ynn[i])==Ytest[i]:
                    countnn = countnn + 1
                if np.argmax(ymat[i])==Ytest[i]:
                    countmat = countmat + 1
    #             for i in range(Xtest.shape[0]):
    #                 print(Ytest[i],ymat[i],ynn[i])
        total = ynn.shape[0]
        if numclasses == 2:
            print float(np.sum((ynn[:,0]>0.5)==Ytest))/total, float(np.sum((ymat[:,0]>0.5)==Ytest))/total
            print roc_auc_score(Ytest,ynn), roc_auc_score(Ytest,ymat),"\n"
        else:
            print countnn/total, countmat/total


def nnfull(Xtrain, Ytrain, Xtest, Ytest,k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtrain, Ytrain[:,0])
    print(k,float(np.sum(neigh.predict(Xtest)==Ytest))/Ytest.shape[0],roc_auc_score(Ytest, neigh.predict_proba(Xtest)[:,1]))


def visualize(xcenters,ycenters, xinit=None, mode='kmeans'):
    import scipy.misc
    import pickle
    # with open(filename,'rb') as infile:
    #     data = pickle.load(infile)
    for i in range(xcenters.shape[0]):
        if mode == 'kmeans':
            scipy.misc.imsave('kmeans'+str(i+1)+'.png', xinit[i,:].reshape((28,28)))
        scipy.misc.imsave('learned'+str(i+1)+'.png', xcenters[i,:].reshape((28,28)))
        print i, ycenters[i,:]


class dataset(object):
    def __init__(self, N, M, D1, D2):
        self.N = N
        self.M = M
        self.D1 = D1
        self.D2 = D2
        self.Z = np.zeros((N,D1), dtype=np.float32)
        self.X = np.zeros((N,D2), dtype=np.float32)
        self.Y = np.zeros((N,1))
        self.C = np.zeros((N,1), dtype=np.float32)
        self.beta = None
        self.mus = None
        self.stds = None
        self.WX = None
        self.WY = None
        self.sigmaX = None
        
    def create(self):
        beta = np.random.dirichlet([1]*self.M)
        mus = np.random.randn(self.M, self.D1)*2
        stds = [[1, 1]]*self.M
        WX = np.random.randn(self.D1, self.D2)
        WY = np.random.randn(self.D1)
        sigmaX = [1,1,1]
        
        for n in range(self.N):
            c = np.argmax(np.random.multinomial(1, beta))
            self.C[n,:] = c
            self.Z[n, :] = np.random.multivariate_normal(mus[c], np.diag(stds[c]))
            self.X[n, :] = np.random.multivariate_normal(np.matmul(self.Z[n],WX), np.diag(sigmaX))
            self.Y[n,:] = np.random.binomial(1,1/(1+np.exp(-np.matmul(self.Z[n],WY))))

        
        self.beta = beta
        self.mus = mus
        self.stds = stds
        self.WX = WX
        self.WY = WY
        self.sigmaX = sigmaX

    def print_params(self):
        print("Cluster Probabilities:", self.beta)
        print("Centers:")
        for i in range(self.M):
            print(self.mus[i,:])
            
        
    def visualize(self):
        color = ['r','g','b','y','c','n','k']
        marker = ['x','+','0']
        for i in [0,1]:
            classpoints = np.where(self.Y==i)
            for j in range(self.M):
                points = np.where(self.C[classpoints]==j)
                Z = self.Z[classpoints[0],:]
                plt.plot(Z[points, 0], Z[points, 1], color[j]+marker[i])
#                 plt.axis([-20,20,-20,20])
                plt.title("Simulated dataset")
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in [0,1]:
            classpoints = np.where(self.Y==i)
            for j in range(self.M):
                points = np.where(self.C[classpoints]==j)
                X = self.X[classpoints[0],:]
                plt.scatter(X[points, 0], X[points, 1], X[points,2],c=color[j], marker=marker[i])
        plt.show()