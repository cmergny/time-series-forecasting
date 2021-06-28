import numpy as np
import matplotlib.pyplot as plt






a = np.random.rand(100)
b = np.random.rand(100)


hist, _, _ = np.histogram2d(a, b, 10)
plt.hist2d(a,b)