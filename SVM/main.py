import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import scipy.io as sio


def pause():
    while True:
        if plt.waitforbuttonpress():
            break


raw_mat = sio.loadmat("SVM/data/ex6data1")
X = raw_mat['X']
y = raw_mat['y']

ypos = y==1
yneg = y==0
plt.ion()
plt.figure()
plt.subplot(131)
plt.scatter(X[:,:1][ypos],X[:,1:][ypos].T ,c='b')
plt.scatter(X[:,:1][yneg],X[:,1:][yneg].T ,c='g')


y[yneg] -= 1

clf = SVC(kernel='linear', C=1)
clf.fit(X,y.ravel())
coef = clf.coef_
intercept = clf.intercept_
w1 = coef[0,0]
w2 = coef[0,1]

xp = np.linspace(0,5,50)
yp = -(w1*xp+intercept)/w2


plt.plot(xp,yp,c='r')
plt.draw()
pause()

raw_mat = sio.loadmat("SVM/data/ex6data2.mat")
X_ex2 = raw_mat['X']
y_ex2 = raw_mat['y']
y_ex2 = y_ex2.ravel()

scaler = StandardScaler()
X_ex2 = scaler.fit_transform(X_ex2)

#print(y_ex2)
C_range = np.logspace(-2,10,5)
gamma_range = np.logspace(-9,3,5)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state=31)
grid = GridSearchCV(SVC(),param_grid=param_grid,cv=cv)

grid.fit(X_ex2, y_ex2)

print("The best parameters are %s with a score of %0.2f" % (grid.best_params_,grid.best_score_))

plt.subplot(132)
plt.scatter(X_ex2[:,0][y_ex2==1],X_ex2[:,1][y_ex2==1] ,c='b')
plt.scatter(X_ex2[:,0][y_ex2==0],X_ex2[:,1][y_ex2==0] ,c='g')

xx, yy = np.meshgrid(np.linspace(min(X_ex2[:,0]),max(X_ex2[:,0]),200), np.linspace(min(X_ex2[:,1]),max(X_ex2[:,1]),200))

Z = grid.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx,yy,Z, colors='r')
plt.draw()
pause()




