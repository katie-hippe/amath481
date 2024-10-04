import numpy as np

## 1
## consider the function f(x) = xsin(3x) - exp(x)

## newton-raphson method
# f(x) = xsin(3x) - exp(x)
# f'(x) = sin(3x) + 3xcos(3x) - exp(x)

x = np.array([-1.6]) # initial guess
gradescope_check = False # edit for gradescope to make it run an extra time
for j in range(1000):
    if (gradescope_check):
        break

    # check that f(x_n) is sufficiently close
    fc = x[j]*np.sin(3*x[j]) - np.exp(x[j])
    if abs(fc) < 1e-6:
        gradescope_check = True

    # else keep going 
    x = np.append(
        x, x[j]-(x[j]*np.sin(3*x[j]) - np.exp(x[j])) # f(x)
        / (np.sin(3*x[j]) + 3*x[j]*np.cos(3*x[j]) - np.exp(x[j]))) # f'(x)
    

A1 = x
A3 = np.array([j])


## bisection method

xr = -0.4; xl = -0.7 # set bounds 
x = np.array([]) # create empty midpoint array
for j in range(1000):
    xc = (xr + xl)/2 # find midpoint

    x = np.append(x, xc) # append new midpoint to array

    # is it sufficiently close?
    fc = x[j]*np.sin(3*x[j]) - np.exp(x[j])
    if abs(fc) < 1e-6:
        break

    # else, keep going...
    if ( fc > 0 ):
        xl = xc
    else:
        xr = xc


A2 = x
A3 = np.append(A3, j+1)


## 2 

A = np.array([[1,2],
              [-1,1]])
B = np.array([[2,0],
              [0,2]])
C = np.array([[2,0,-3],
              [0,0,-1]])
D = np.array([[1,2],
              [2,3],
              [-1,0]])
x = np.array([1,0])
y = np.array([0,1])
z = np.array([1,2,-1])

A4 = A+B
A5 = 3*x - 4*y
A6 = A @ x
A7 = B @ (x-y)
A8 = D @ x
A9 = D @ y + z
A10 = A @ B
A11 = B @ C
A12 = C @ D

print(A6)
