import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg


def pause():
    while True:
        if plt.waitforbuttonpress():
            break


def KmeansInitCentroids(X,K):
    randidx = np.random.permutation(X.shape[0])
    init_centroids = X[randidx[0:K],:]
    return init_centroids

def findClosestcentroids(X, centroids):
    K = centroids.shape[0]
    m = X.shape[0]
    dist_mtx = np.zeros((m,K))
    for i in range(K):
        diff = X-centroids[i]
        diff_sq = diff**2
        dist_mtx[:,i] = diff_sq.sum(axis=1)
    idx = dist_mtx.argmin(axis=1)
    return idx

def computeCentroids(X, idx, K):
    m,n = X.shape
    centroids = np.zeros((K, n))
    for i in range(K):
        X_temp = X[idx==i,:]
        centroids[i,:] = X_temp.mean(axis=0)
    return centroids

def plotDataPoint(X, idx, K):
    colors = iter(cm.rainbow(np.linspace(0,1,K)))
    for i in range(K):
        plt.scatter(X[idx==i,0],X[idx==i,1],color=next(colors))
    plt.draw()

def plotLine(p1,p2):
    plt.plot(np.c_[p1[0],p2[0]].flatten(),np.c_[p1[1], p2[1]].flatten(),c='black',linewidth=2.0 ,marker='d')

def plotProgressKmeans(X,centroids,previous_centorids, idx, K, iter):
    plotDataPoint(X, idx, K)
    plt.scatter(centroids[:,0],centroids[:,1],c='black')
    for i in range(K):
        plotLine(centroids[i,:],previous_centorids[i,:])
        plt.draw()
    plt.title("iteration: %d" %iter)

def runKmeans(X, initialcentorids, max_iter=10, plot_progress=False):
    m,n = X.shape
    K = initialcentorids.shape[0]
    centroids = initialcentorids
    previous_centroids = centroids
    idx = np.zeros((m,1))
    if plot_progress and n == 2:
        plt.ion()
        plt.figure()

    for i in range(max_iter):
        idx = findClosestcentroids(X,centroids)

        if plot_progress:
            plotProgressKmeans(X,centroids,previous_centroids,idx,K,i+1)
            previous_centroids = centroids
            pause()
        centroids = computeCentroids(X, idx, K)

    return centroids, idx

def evaluateKmeans():
    pass


if __name__ == '__main__':

    raw_data = sio.loadmat("K-Means/data/ex7data2.mat")
    X = raw_data['X']

    K = 3
    max_iter = 10

    init = np.array([3,3,6,2,8,5]).reshape(3,2)

    centroids, idx = runKmeans(X,init,max_iter,True)


    '''
    image compression
    '''
    plt.clf()


    image = mpimg.imread("K-Means/data/bird_small.png")

    h, w , dummy = image.shape
    X_png = image.reshape(h*w,3)

    K_png = 16
    init_png = KmeansInitCentroids(X_png,K_png)

    centroids_png, idx_png = runKmeans(X_png, init_png, 10)

    new_image = centroids_png[idx_png, :]
    new_image = new_image.reshape(h,w,3)


    plt.subplot(121)
    plt.title("Original")
    plt.imshow(image)

    plt.subplot(122)
    plt.title("Compressed version")
    plt.imshow(new_image)
    pause()