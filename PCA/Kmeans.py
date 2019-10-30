import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pause as ps
import plotLine as plotL



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


def plotProgressKmeans(X,centroids,previous_centorids, idx, K, iter):
    plotDataPoint(X, idx, K)
    plt.scatter(centroids[:,0],centroids[:,1],c='black')
    for i in range(K):
        plotL.plotLine(centroids[i,:],previous_centorids[i,:])
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
            ps.pause()
        centroids = computeCentroids(X, idx, K)

    return centroids, idx