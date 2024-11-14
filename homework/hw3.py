import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import math
from scipy.special import hermite




##### part a ##### 

tol = 1e-6 # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm'] # eigenfunc colors

n0 = 1; A = 1 
y0 = [0, A] # initial conditions? need to edit these 
L=4; xp = [-L, L]; dx = .1

xspan = np.linspace(xp[0], xp[1], 20 * L + 1)

# set up variables
A1 = np.empty((20 * L + 1, 0))
A2 = []

def  shoot_a(x, y, n0, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

epsilon_start = 0.2 # beginning value of epsilon
for modes in range(1, 6): # begin mode loop
    epsilon = epsilon_start # initial value of eigenvalue epsilon
    depsilon = n0 / 100 # default step size in epsilon

    y0 = [1, np.sqrt(L**2 - epsilon)] # left hand side!  

    for _ in range(1000): # begin convergence loop for epsilon
        sol = solve_ivp(shoot_a, [-L, L+dx], y0, t_eval=xspan, args=(n0,epsilon))
        y = sol.y.T

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
    norm = np.trapezoid(y[:, 0] * y[:, 0], xspan) # calculate the normalization
    normed = y[:, 0] / np.sqrt(norm)
    #plt.plot(xspan, normed, col[modes - 1], 
            # label="\u03A6" + str(modes)) # plot modes
    
    # update our function matrix
    A1 = np.c_[A1,abs(normed)]

#plt.legend()
#plt.show()

print("A1:", A1)
print("A2:",A2)



#### part b ####

L = 4
xspan = np.linspace(xp[0], xp[1], 20 * L + 1)
N = len(xspan)
dx = xspan[1] - xspan[0]

A3 = np.empty((20 * L + 1, 0)) # initialize our A3

B = np.zeros((N-2, N-2)) # create our matrix! 

# fill up diagonals 
for j in range(N-2): 
    B[j, j] = -2 - dx**2 * (xspan[j+1]**2)
for j in range(N-3):
    B[j, j + 1] = 1
    B[j + 1, j] = 1

B[0, 0] += 4/3
B[0, 1] -= 1 / 3
B[N - 3, N - 3] += 4/3
B[N - 3, N - 4] -= 1/3


# now compute the eigenvalues and vectors!
eigval, eigvec = eigs(-B, k=5, which="SM") # eigvec has interior points

# append the boundary conditions

phi_0 = np.array([(4/3) * eigvec[0,:] - (1/3) * eigvec[1,:]])
phi_n = np.array([(4/3) * eigvec[-1,:] - (1/3) * eigvec[-2,:]])
eigvec = np.vstack((phi_0,eigvec,phi_n))


eigval = np.real(eigval) / (dx**2)


for modes in range(5):

    norm = np.trapezoid(eigvec[:,modes] * eigvec[:, modes], xspan) # calculate the normalization
    normed = eigvec[:, modes] / np.sqrt(norm)


    #plt.plot(xspan, normed, col[modes], 
             #label="\u03A6" + str(modes)) # plot modes

    A3 = np.c_[A3,abs(normed)]

#plt.legend()
#plt.show()

A4 = eigval

print("A3:", A3)
print("A4:", A4)





#### part c 

# set up variables
L=2; K = 1; dx = 0.1
x_span = np.arange(-L, L+dx,dx)
tol = 1e-6


# positive gamma 
A5 = np.empty((20 * L + 1, 0))
A6 = []

# negative gamme
A7 = np.empty((20 * L + 1, 0))
A8 = []

# define shoot function
def shoot_c(x,y,epsilon,gamma):
    return [y[1], (gamma * np.abs(y[0])*np.abs(y[0]) + K * x**2 - epsilon) * y[0]]


# do the thing
for gamma in [0.05, -0.05] : # gamma loop
    # set up variables 
    e0 = 0.1
    A = 1e-6

    # modes loop
    for mode in range(1, 3):
        da = 0.01

        # A loop
        for i in range(1000):
            epsilon = e0
            depsilon = 0.2

            # epsilon loop
            for _ in range(1000):
                y0 = [A, A * np.sqrt(L**2 - epsilon)]
                sol = solve_ivp(shoot_c, [-L, L + dx], y0, t_eval=x_span, args=(epsilon, gamma))
                yS = sol.y.T
                xS = sol.t

                # define and check boundary conditions
                bc = yS[-1, 1] + np.sqrt(L**2 - epsilon) * yS[-1, 0]
                if abs(bc) < tol: # if we're good, we're done!
                    break
                if (-1)**(mode + 1)*bc > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2

            # now check the area 
            area = simpson(yS[:, 0]**2, x=xS)
            if abs(area-1) < tol: # if we're within tolerance, we're good!
                #print(f"Gamma {gamma}, Mode {mode} area converged at {i} iterations")
                break
            if area < 1:
                A += da
            else:
                A -= da/2
                da /= 2
        e0 = epsilon + 0.2

        if gamma > 0:
            # save pos values

            A5 = np.c_[A5,abs(yS[:,0])]
            A6.append(epsilon)
        
        else:
           # save negative values 

            A7 = np.c_[A7, abs(yS[:,0])]
            A8.append(epsilon)


print("A5:", A5)
print("A6:",A6)
print("A7:",A7)
print("A8:",A8)
            

## part d


L = 2
K = 1
E = 1  # fixed energy value
gamma = 0
x_span = [-L, L]
tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# initial conditions
phi0 = 1
phi_x0 = np.sqrt(K * L**2 - E)
y0 = [phi0, phi_x0]

# right-hand side of the ODE
def hw1_rhs_a(x, y, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

# store results
slopes = []

# function to perform convergence study for each method
def run_convergence_study(method_name, order):
    avg_step_sizes = []
    for tol in tolerances:
        options = {'rtol': tol, 'atol': tol}
        sol = solve_ivp(hw1_rhs_a, x_span, y0, method=method_name, args=(E,), **options)
        
        # calculate average step size
        step_sizes = np.diff(sol.t)
        avg_step_size = np.mean(step_sizes)
        avg_step_sizes.append(avg_step_size)

    # log-log plot and slope calculation
    log_tolerances = np.log10(tolerances)
    log_avg_step_sizes = np.log10(avg_step_sizes)
    #plt.plot(log_avg_step_sizes, log_tolerances, label=f'{method_name}')
    
    # use polyfit to find the slope
    slope, _ = np.polyfit(log_avg_step_sizes, log_tolerances, 1)
    slopes.append(slope)

    
# run convergence study for each method
run_convergence_study('RK45', order=4)  # RK45 method (4th order)
run_convergence_study('RK23', order=3)  # RK23 method (2nd order)
run_convergence_study('Radau', order=5) # Radau method (5th order implicit)
run_convergence_study('BDF', order=3)   # BDF method (3rd order implicit)

# Show the plot
#plt.legend()
#plt.show()


A9 = np.array(slopes).reshape((4,)).flatten()
print("A9:",A9)





## part e

# first define our own factorial function
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

L = 4
x = np.linspace(-L,L, 20*L+1)

# this is comparing (a) and (b) answers with online answers


phi = np.empty((81, 0))

def hermite_func(n,x):
    if n == 0:
        return np.exp(-x**2 / 2) / np.pi**0.25
    elif n == 1:
        return np.sqrt(2) * x * np.exp(-x**2 / 2) / np.pi**0.25
    elif n == 2:
        return (2 * x**2 - 1) * np.exp(-x**2 / 2) / (np.sqrt(2) * np.pi**0.25)
    elif n == 3:
        return (2 * x**3 - 3 * x) * np.exp(-x**2 / 2) / (np.sqrt(3) * np.pi**0.25)
    elif n == 4:
        return (4 * x**4 - 12 * x**2 + 3) * np.exp(-x**2 / 2) / (2 * np.sqrt(6) * np.pi**0.25)


for j in range(5):

    temp = hermite_func(j,x)

    norm = np.trapezoid(temp * temp, x) # calculate the normalization
    normed = temp / np.sqrt(norm)
    
    phi = np.c_[phi, normed]

    plt.plot(x, normed, col[modes - 1], 
             label="\u03A6" + str(modes)) # plot modes

plt.legend()
plt.show()
# first calculate function error
# this will look like abs(ours - actual), then take the area under (that curve squared)

erphi_a = np.zeros(5)
erphi_b = np.zeros(5)
er_a = np.zeros(5)
er_b = np.zeros(5)

for j in range(5):
    erphi_a[j] = simpson((abs(A1[:,j]) - abs(phi[:,j]))**2, x=x)
    erphi_b[j] = simpson((abs(A3[:,j]) - abs(phi[:,j]))**2, x=x)


    # then value error (simple percent error)   
    er_a[j] = 100 * abs(A2[j] - (2*j + 1)) / (2*j + 1)
    er_b[j] = 100 * abs(A4[j] - (2*j + 1)) / (2*j + 1)

A10 = erphi_a
A12 = erphi_b

A11 = er_a
A13 = er_b

print("A10:", A10)
print("A11:", A11)
print("A12:", A12)
print("A13:", A13)