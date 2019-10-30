import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as op
import os


def Wrapper(R,y,nu,nm,nf,lamb,grad=True):

    def cofiCostFunction(params,debug = False):
        params = params.ravel()
        X = params[:(nm * nf)].reshape(-1,1)
        Theta = params[(nm*nf):].reshape(-1,1)
        if debug == True:
            for i in range(100):
                print("%d %d   "%(X[i],Theta[i]),end='')
        Xre = X.reshape(nm,nf)
        Tre = Theta.reshape(nu,nf)
        yre = y.reshape(nm,nu)
        P = np.dot(Xre,Tre.T ) * R
        diff = P - yre
        J = 0.5 * np.sum(diff ** 2)
        J += 0.5 * lamb * (np.dot(X.T,X) + np.dot(Theta.T, Theta))
        if not grad:
            return J
        '''
        여기서 주의해야 할 점이 unrolling할때 grad_x 는 (num_m,num_f)여야하고
        grad_theta는 (num_u,num_f)여야한다는 것이다
        '''
        Grad_theta = np.dot(Xre.T ,diff).T.reshape(-1,1) + lamb * Theta
        Grad_X =  np.dot(Tre.T, diff.T).T.reshape(-1,1) + lamb * X
        Grad = np.concatenate((np.ravel(Grad_X), np.ravel(Grad_theta)))
        return J,Grad
    return cofiCostFunction


def normalizeRatings(y,R):
    m,n = y.shape
    ymean = np.zeros((m,1))
    ynorm = np.zeros((m,n))
    for i in range(m):
        idx = (R[i,:] != 0)
        ymean[i] = y[i,:][idx].mean()
        ynorm[i,:][idx] = y[i,:][idx] - ymean[i]
    return ynorm, ymean

def checkCostFucntion(lamb=0):

    X = np.random.rand(12).reshape(4,3)
    Theta = np.random.rand(15).reshape(5,3)
    Y = np.dot(X, Theta.T)
    m,n = Y.shape
    Y[np.random.rand(m*n).reshape(m,n) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[ Y != 0 ] = 1

    cofi = Wrapper(R,Y,5,4,3,lamb,True)
    check = Wrapper(R,Y,5,4,3,lamb,False)
    params = np.vstack((X.reshape(-1,1),Theta.reshape(-1,1)))
    J, grad = cofi(params)

    numgrad = np.zeros(params.shape)
    perturb = np.zeros(params.shape)
    e = 1e-4
    for p in range(np.size(params)):
        perturb[p] = e
        loss1 = check(params+perturb)
        loss2 = check(params-perturb)

        numgrad[p] = (loss1 - loss2) / (2*e)

        perturb[p] = 0

    for i in range(grad.shape[0]):
        print("%f  %f" %(grad[i], numgrad[i]))
    print("Left is gradient and  Right is numerical gradient...")
    #diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)



def myRating(myrating, total=1682):
    nm = int(input("enter the number of movie you want to rate"))
    movielist = getMovielist()
    i = 0
    while i < nm:
        print("%d: " % (i+1), end='')
        num, rat = map(int, input("# of Movie, Your Rating").split())
        print("%d-%s: %d [y/n]?" %(num+1, movielist[num], rat))
        a = input()
        while a != 'y' and a != 'Y' and a != 'n' and a != 'N':
            print("you must answer in a form of y or n")
            a = input()

        if a == 'y' or a == 'Y':
            i += 1
            myrating[i] = rat
        else:
            print("you need to try again")
            continue
    return myrating

def getMovielist():
    movielist = []
    f = open("CollaborateFiltering/Numpy_ver/data/movie_ids.txt")
    i=0
    while True:
        #print(i,end='')
        line = f.readline()
        line = line.rstrip()
        if line == '':
            break
        num, mov = line.split(' ',1)
        movielist.append(mov)
        i +=1
    return movielist

def paramsInit(num_movies, num_users, num_featrues):
    X = np.random.random(size=(num_movies , num_featrues))
    Theta = np.random.random(size=(num_users , num_featrues))

    return np.concatenate((X.ravel(),Theta.ravel()))

raw_data = sio.loadmat("CollaborateFiltering/Numpy_ver/data/ex8_movies.mat")
Y = raw_data['Y']
R = raw_data['R']
raw_data = sio.loadmat("CollaborateFiltering/Numpy_ver/data/ex8_movieParams.mat")
#print(raw_data.keys())
X = raw_data['X']
Theta = raw_data['Theta']


num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies,:num_features]
Theta = Theta[:num_users,:num_features]
Y = Y[:num_movies,:num_users]
R = R[:num_movies,:num_users]

J = Wrapper(R,Y,num_users,num_movies,num_features,0,False)

cost = J(np.vstack((X.reshape(-1,1),Theta.reshape(-1,1))))
print("22.22: %.2f"%cost)


J = Wrapper(R,Y,num_users,num_movies,num_features,1.5,True)

cost,grad = J(np.vstack((X.reshape(-1,1),Theta.reshape(-1,1))))
print("31.34: %.f"%cost)
#print(grad)
checkCostFucntion(1.5)

movieList = getMovielist()
print(len(movieList))
my_ratings = np.zeros(len(movieList)).reshape(-1,1)
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5
#my_ratings = myRating(my_ratings)

for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated %d for %s" %(my_ratings[i],movieList[i]))


raw_data = sio.loadmat("CollaborateFiltering/Numpy_ver/data/ex8_movies.mat")
Y = raw_data['Y']
R = raw_data['R']
Y = np.append(Y,my_ratings,axis=1)
R = np.append(R,my_ratings !=0, axis=1)
Ynorm, Ymean = normalizeRatings(Y,R)

num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10
params = paramsInit(num_movies,num_users,num_features)

lamb = 10

CostFunc = Wrapper(R,Ynorm,num_users,num_movies,num_features,lamb,True)

fmin = op.minimize(fun=CostFunc, x0=params, method='CG', jac=True, options={'maxiter': 100})
X = fmin.x[:num_movies * num_features].reshape(num_movies, num_features)
Theta = fmin.x[num_movies * num_features:].reshape(num_users, num_features)
print(fmin)
p = np.dot(X, Theta.T)
my_predictions = p[:,-1].ravel() + Ymean.ravel()



ix = np.argsort(my_predictions,axis=0)[::-1]
for i in range(10):
    j = int(ix[i])
    print("Predicting rating %.1f for movie %s" % (my_predictions[j], movieList[j]))



mode = 'b'
if mode == 'b' or mode == 'B':
    i = 0
    print("START")
    alpha = 1e-3
    pre = 1e7
    while i < 1000:
        Cost, Grad = CostFunc(params)

        if i % 100 == 0:
            print(Cost/np.sum(R))
        params -= alpha * Grad
        i += 1

    X = params[:num_movies * num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)

    p = np.dot(X, Theta.T)
    p += Ymean

    my_predictions = p[:,-1]

    ix = np.argsort(my_predictions,axis=0)[::-1]
    for i in range(10):
        j = ix[i]
        print("Predicting rating %.1f for movie %s" %(my_predictions[j], movieList[j]))

elif mode == 's' or mode == 'S':
    pass
