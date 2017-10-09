import os
import struct
import numpy as np

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