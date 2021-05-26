import numpy as np
import matplotlib.pyplot as plt


def MutualInfo(X, Y, bins):
    """ Computes the mutal information bw two time series """
    # numpy histo returns a tuple (hist, bins)
    hist_XY = np.histogram2d(X,Y,bins)[0]
    hist_X = np.histogram(X,bins)[0]
    hist_Y = np.histogram(Y,bins)[0]

    H_X = Entropy(hist_X)
    H_Y = Entropy(hist_Y)
    H_XY = Entropy(hist_XY)
    return(H_X+H_Y-H_XY)

def Entropy(hist):
    # Compute proba from histo
    p = hist / float(np.sum(hist))
    # Remove all zeros values
    p = p[p>0]
    # Compute Entropy
    H = -sum(p* np.log2(p))  
    return(H)




hist = np.histogram2d(a, b, 10)