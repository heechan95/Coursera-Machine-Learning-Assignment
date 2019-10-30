import numpy as np
import matplotlib.pyplot as plt


def plotLine(p1,p2,str_opt,kwargs):
    if str_opt:
        plt.plot(np.c_[p1[0], p2[0]].flatten(), np.c_[p1[1], p2[1]].flatten(), str_opt,**kwargs)
    else:
        if kwargs:
            plt.plot(np.c_[p1[0], p2[0]].flatten(), np.c_[p1[1], p2[1]].flatten(),**kwargs)
        else:
            plt.plot(np.c_[p1[0],p2[0]].flatten(),np.c_[p1[1], p2[1]].flatten(),c='black',linewidth=2.0 ,marker='d')