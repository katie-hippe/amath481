import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags


## construct matrix A 

m = 8    # N value in x and y directions
n = m * m  # total size of matrix
L = 10 # scope of x and y
dx = (2*L) / m # dx and dy because they're the same

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

matA = spdiags(diagonals, offsets, n, n).toarray()

# Plot matrix structure
fig, ax = plt.subplots()
cax = ax.matshow(matA)
fig.colorbar(cax)
plt.show()

A1 = matA / (dx**2)





## construct matrix b

diagonals = [e1.flatten(),-1*e1.flatten(), e1.flatten(),
             -1*e1.flatten()]
offsets = [-(n-m),-m, m, n-m]

matB = spdiags(diagonals, offsets, n, n).toarray()

# Plot matrix structure
fig, ax = plt.subplots()
cax = ax.matshow(matB)
fig.colorbar(cax)
plt.show()

A2 = matB / (2*dx)




## construct matrix c


diagonals = [ e5.flatten(), 
             -1*e2.flatten(), e3.flatten(), 
             -1*e4.flatten() ]
offsets = [ -m+1, -1,  1, m-1]

matC = spdiags(diagonals, offsets, n, n).toarray()

fig, ax = plt.subplots()
cax = ax.matshow(matC)
fig.colorbar(cax)
plt.show()

A3 = matC / (dx*2)