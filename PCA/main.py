import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.image as mpimg
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import Kmeans as km
import plotLine as pL
from display import displayData


def pause():
    while True:
        if plt.waitforbuttonpress():
            break


def pca(X):
    m = X.shape[0]
    sigma = np.dot(X.T,X)
    sigma /= m
    U, S ,V = np.linalg.svd(sigma)
    return U, S

def projectData(X,U,K):
    U_k = U[:,0:K]
    Z = np.dot(U_k.T, X.T)
    return Z.T

def recoverData(Z,U,K):
    U_k = U[:,0:K]
    X_rec = np.dot(U_k ,Z.T)
    return X_rec.T



if __name__ == '__main__':

    raw_data = sio.loadmat("PCA/data/ex7data1.mat")
    X = raw_data['X']

    plt.ion()
    plt.figure()
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    plt.axis([-4,3,-4,3])
    plt.scatter(X_norm[:,0],X_norm[:,1], facecolors='none', edgecolors='b')
    pause()


    K=1
    U, S = pca(X_norm)
    Z = projectData(X_norm,U,K)
    X_rec = recoverData(Z, U, K)
    plt.scatter(X_rec[:,0],X_rec[:,1], facecolors='none', edgecolors='r')
    pause()

    options = {'linewidth':1.0}

    for i in range(X_norm.shape[0]):
        pL.plotLine(X_norm[i,:],X_rec[i,:],'--k',options)

    pause()
    plt.clf()
    raw_data = sio.loadmat("PCA/data/ex7faces.mat")
    X = raw_data['X']

    displayData(X[0:100,:])
    pause()

    X_norm = scaler.fit_transform(X)
    U, S = pca(X_norm)

    displayData(U[:,0:36].T)
    pause()
    K = 100
    Z = projectData(X_norm, U, K)
    X_rec = recoverData(Z,U,K)

    plt.clf()
    plt.close()

    plt.figure(figsize=(8,5))
    plt.subplot(121)
    plt.title("Original")
    displayData(X[0:100, :])

    plt.subplot(122)
    plt.title('Reconstructed version')
    displayData(X_rec[0:100,:])
    pause()


    A = mpimg.imread("PCA/data/bird_small.png")
    A /= 255
    K = 16
    h, w , dummy = A.shape
    A = A.reshape(-1,3)
    centroids, idx = km.runKmeans(A,km.KmeansInitCentroids(A,K))
    sel = np.random.randint(0,h*w-1,1000)
    colors = iter(cm.rainbow(np.linspace(0,1,K+1)))

    plt.clf()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(K):
        A_new = (A[sel,:])[idx[sel] == i,:]
        ax.scatter(A_new[:,0],A_new[:,1],A_new[:,2], s=1.0, color=next(colors))
    pause()

    ax = plt.subplot(111)
    A_norm = scaler.fit_transform(A)
    U, S = pca(A)
    Z = projectData(A,U,2)
    colors = iter(cm.rainbow(np.linspace(0,1,K+1)))
    for i in range(K):
        Z_new = (Z[sel,:])[idx[sel] == i,:]
        ax.scatter(Z_new[:,0],Z_new[:,1], s=1.0, facecolors='none', edgecolor=next(colors))
    pause()
