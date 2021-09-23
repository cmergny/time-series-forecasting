# Numerical simulation of Harmonic Oscillators
# Generates the spring_data.txt file
# This zrticifial data is used to validate
# the neural networks.


### IMPORTS

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

### FUNCTIONS

def SpringsODE(X, t, M, R):
    """Defines the fst order diff equations 
    for the spring-mass system.
    Ex: y1 = X[0]    x1 = X[3]
        y2 = X[1]    x2 = X[4]
        y3 = X[2]    x3 = X[5]
    F = (y1', y2', y3', x1', x2', x3') 
    """
    # Parameters
    k = 1
    n = len(X)//2
    a = 1
    # Return Matrix
    F = [0 for k in range(2*n)]
    
    # Intermediate spings
    for i in range(n):
        # F = Frappel -Ffrottements -Coupling
        F[i] = - k/M[i]*(X[i+n]) #- b*X[i] - a*k/M[i]*np.dot(R[i], X[:n])
    
    # Assign the derivatives
    for i in range(n):
        F[i+n] = X[i]
    return F

def SimulateSpings(n, X=None, plot=True):
    """Run the ODE solver for the couple springs system,
    Plot and returns xn(t) for all n, where xn is the motion 
    of the nth spring.
    """
    # Initial conditions
    Mi = [np.random.randint(1, 8) for k in range(n)]
    Xi = [np.random.randint(1, 8) for k in range(n)]
    X = [0 for k in range(n)] + Xi if X is None else X
    # Coupling matrice
    R = np.zeros((n, n)) # no coupling
    # time
    stoptime = 20000
    t = np.linspace(0, stoptime, 2*stoptime)
    # Call the ODE solver.
    sol = odeint(SpringsODE, X, t, args=(Mi, R))
    data = sol[:, n:]
    # Plot
    if plot:
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot(data[:, i], label = f'spring {i}', alpha=0.5)
        plt.legend()
        plt.show()
    return(data)

def AnimateSprings(data):
    """ Animate the mvt of coupled spings with dots"""
    n = len(data[0, :]) # nbr of spings
    # Animation fig
    fig = plt.figure()
    ax = plt.axes(xlim=(-10, 100), ylim=(-20, 1))
    lines = [ax.plot([], [],'o')[0] for k in range(n)]
    
    def init():
        """Init function"""
        for line in lines:
            line.set_data([], [])
        return line,
    
    def animate(i):
        """Updates each frame"""
        j = 0
        for line in lines:
            x = data[i, j]  # seperate each sping
            y = 0 - j*2
            j += 1
            line.set_data(x, y)
        return lines
    
    # run the animation
    ani = FuncAnimation(fig, animate, init_func=init, frames=len(data[:,0]), interval=30, blit=True)
    plt.show()

def WriteSimulation(data, filename='data/spring_data.txt'):
    """Write the data array into a txt file"""
    with open(filename, 'w') as f:
        for i in range(len(data[:,0])):
            text = ''
            for j in range(len(data[0, :])):
                text += str(data[i, j]) + '\t'
            text += '\n'
            f.write(text)
    print(f"Array of springs motion written in {filename}")
    
if __name__ == '__main__':

    n = 10 # nbr of springs
    data = SimulateSpings(n, plot=False) # array of xn(t)
    AnimateSprings(data) # Animation of motion
    WriteSimulation(data)

        
        

