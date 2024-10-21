import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math


tol = 1e-6 # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm'] # eigenfunc colors

n0 = 1; A = 1 
y0 = [0, A] # initial conditions? need to edit these 
L=4; xp = [-L, L]

xshoot = np.linspace(xp[0], xp[1], 20 * L + 1)

# set up variables
A1 = np.empty((20 * L + 1, 0))
A2 = []

def shoot(y, x, n0, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

epsilon_start = 0.2 # beginning value of epsilon
for modes in range(1, 6): # begin mode loop
    epsilon = epsilon_start # initial value of eigenvalue epsilon
    depsilon = n0 / 100 # default step size in epsilon

    y0 = [1, np.sqrt(L**2 - epsilon)] # left hand side!  

    for _ in range(1000): # begin convergence loop for epsilon
        y = odeint(shoot, y0, xshoot, args=(n0,epsilon))

        # check for convergence
        if abs((y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1,0]) - 0) < tol:
            A2.append(epsilon) # add eigenvalue to list
            break # get out of convergence loop
        
        if (-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1,0]) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2

    epsilon_start = epsilon + 2 # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xshoot) # calculate the normalization
    normed = y[:, 0] / np.sqrt(norm)
    plt.plot(xshoot, normed, col[modes - 1], 
             label="\u03A6" + str(modes)) # plot modes
    
    # update our function matrix
    A1 = np.c_[A1,abs(normed)]

plt.legend()
plt.show() 

print(A1)
print(A2)