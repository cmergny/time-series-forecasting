import numpy as np
import matplotlib.pyplot as plt
### INTRASIC DIM ESTIMATORS

def PCA(data, thresh=0.5):
    """
    Return estimation of dataset intrasic dimension using
    Principal Component Analysis
    """
    R = np.corrcoef(data.transpose())
    values, vectors = np.linalg.eig(R)
    values = values / np.max(values)
    print("id = ", len(values[values > thresh]))
    return values


def MaxLikelihood(data, k=3):
    """
    Return estimation of dataset intrasic dimension using
    article: Elizaveta Levina, Peter J. Bickel
    Maximum Likelihood Estimation of Intrinsic Dimension"""
    # k is nbr of neirest neigbhors to take into account
    n = data.shape[1]
    Matrix = np.zeros((n, n))
    # Compute Euclidian distance
    for i in range(n):
        for j in range(n):
            Matrix[i, j] = np.linalg.norm(data[:, i] - data[:, j])
    Matrix.sort()  # sort by Nearest Neigh
    mk = 0  # computed dimension
    for x in range(n):
        mkx = 0
        for j in range(1, k):  # to k-1
            mkx += np.log(Matrix[x, k] / Matrix[x, j])
        mk += (1 / (k - 1) * mkx) ** (-1)
    mk = mk / n
    return mk


def FourierTransform(data, plot=True):
    """Return the fourier spectrum of the data"""
    if len(data.shape) < 2:
        data = np.expand_dims(data, 1)
    transforms = np.zeros(data.shape)
    for i in range(data.shape[1]):
        transforms[:, i] = np.abs(np.fft.fft(data[:, i]))
    transforms = transforms[: transforms.shape[0] // 2, :]
    if plot:
        plt.plot(transforms)
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
    return transforms


### MUTUAL INFORMATION


def MutualMatrix(data, bins=50):
    """Computes mutual info matrix
    # todo : verfier code similaire à scipy
    """

    def MutualInfo(X, Y, bins):
        """Computes the mutal information bw two time series
        I(X, Y) = H(x) + H(Y) - H(X, Y) where H is shannon entropy"""
        # Compute histogram to get probabilities
        # numpy histo returns a tuple (hist, bins)
        hist_XY = np.histogram2d(X, Y, bins)[0]
        hist_X = np.histogram(X, bins)[0]
        hist_Y = np.histogram(Y, bins)[0]
        # Compute Entropy
        H_X = Entropy(hist_X)
        H_Y = Entropy(hist_Y)
        H_XY = Entropy(hist_XY)
        # Return Mutual Info
        return H_X + H_Y - H_XY

    def Entropy(hist):
        """Computes Shannon entropy from histogramm"""
        # Compute proba from histo
        p = hist / float(np.sum(hist))
        # Remove all zeros values
        # p*log(p) -> 0 when p->0
        p = p[p > 0]
        # Compute Entropy
        H = -sum(p * np.log2(p))
        return H

    # Matrix of all pairwise mutual info
    n = data.shape[1]  # nbr modes
    Matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Matrix[i, j] = MutualInfo(data[:, i], data[:, j], bins=bins)
        Matrix[i, i] = 2.6
    # plt.pcolormesh(M)
    return Matrix


def AirQualityData(width=500):
    """import the air quality file dataset"""
    data = np.zeros((width, 13 - 2))
    df = pd.read_csv("Data/air_quality.csv", sep=";", index_col=False)
    # import wanted columns
    for j, column in enumerate(df.columns[2:13]):
        ar = np.array(df[column][:width])
        # convert str to float
        for i, elt in enumerate(ar):
            if type(elt) == str:
                ar[i] = float(elt.replace(",", "."))
            ar[i] = 0 if ar[i] < -100 else ar[i]
    return data


# scp mergny@grappe:/workdir/mergny/OUTPUT/time_coefficients.d ~/Documents/Work/PLR2/LSTM_mergny/LSTM_build/Data/time_coefficient.d

#%%
### MAIN
if __name__ == "__main__":
    pass