import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import spdiags, csr_matrix
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import lu, solve_triangular
from scipy.sparse.linalg import gmres, spsolve, bicgstab
import time

# MASTER PARAMETERS
n_master = 64

## construct matrix A 

m = n_master    # N value in x and y directions
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

matA = spdiags(diagonals, offsets, n, n) / (dx**2)

# Plot matrix structure
#fig, ax = plt.subplots()
#cax = ax.matshow(matA)
#fig.colorbar(cax)
#plt.show()






## construct matrix b

diagonals = [e1.flatten(),-1*e1.flatten(), e1.flatten(),
             -1*e1.flatten()]
offsets = [-(n-m),-m, m, n-m]

matB = spdiags(diagonals, offsets, n, n) / (2*dx)

# Plot matrix structure
#fig, ax = plt.subplots()
#cax = ax.matshow(matB)
#fig.colorbar(cax)
#plt.show()





## construct matrix c


diagonals = [ e5.flatten(), 
             -1*e2.flatten(), e3.flatten(), 
             -1*e4.flatten() ]
offsets = [ -m+1, -1,  1, m-1]

matC = spdiags(diagonals, offsets, n, n) / (dx*2)

#fig, ax = plt.subplots()
#cax = ax.matshow(matC)
#fig.colorbar(cax)
#plt.show()






#### begin homework 5 material here ###########

# first let's set our intial conditions

# Define parameters
tspan = np.arange(0, 4.5, .5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = n_master, n_master
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)
w0 = 1 * np.exp(-X**2 - .05 * Y**2) #+ 1j * np.zeros((nx, ny))  # Initialize as complex

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6 # reset initial value! 
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6 # for both x and y! 
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# define the ODE system 
def a_rhs(t, w, nx, ny, N,K, nu, A, B, C):
    # first get psi
    wt = fft2(w.reshape(nx, nx)) # get the correct transform 
    psi = np.real(ifft2(-wt / K)).flatten() # now we've gotten our psi!

    # now put it all together! 
    rhs = (nu*A.dot(w) - (B.dot(psi)) * (C.dot(w)) + (C.dot(psi)) * (B.dot(w)))

    return rhs


# Solve the ODE and plot the results

# start our timer
start_time = time.time() # Record the start time

# solve! 
sol = solve_ivp(a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", 
                t_eval=tspan, args=(nx, ny, N, K, nu, matA, matB, matC))
sol = sol.y.reshape(nx, nx, len(tspan))

# how long did that take?
end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"FFT Elapsed Time: {elapsed_time:.2f} seconds")


split = 1
for j, t in enumerate(tspan):
    if (j % split == 0):
        w = sol[:, :, j]
        plt.subplot(3, 3, j//split + 1)
        plt.pcolor(x, y, w, shading='auto')
        plt.title(f'Time: {t}')
        plt.colorbar()

plt.tight_layout()
plt.show()

A1 = sol.reshape(N, len(tspan))

print(A1.shape)


## part b 

# redefine matrix A 

b_matA = csr_matrix(matA)
b_matA[0,0] = 2



# first we try direct solve
# x = np.linalg.solve(A, b)

def b_direct_rhs(t, w, nx, ny, N,K, nu, A, B, C):
    # first get psi
    psi = spsolve(A,w)

    # now put it all together! 
    rhs = (nu*A.dot(w) - (B.dot(psi)) * (C.dot(w)) + (C.dot(psi)) * (B.dot(w)))

    return rhs

# start our timer
# start_time = time.time() # Record the start time

# # solve! 
sol = solve_ivp(b_direct_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", 
                t_eval=tspan, args=(nx, ny, N, K, nu, b_matA, matB, matC))
sol = sol.y.reshape(nx, nx, len(tspan))

# how long did that take?
end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"A\\b Elapsed Time: {elapsed_time:.2f} seconds")



split = 1
for j, t in enumerate(tspan):
    if (j % split == 0):
        w = sol[:, :, j]
        plt.subplot(3, 3, j//split + 1)
        plt.pcolor(x, y, w, shading='auto')
        plt.title(f'Time: {t}')
        plt.colorbar()

plt.tight_layout()
plt.show()

A2 = sol.reshape(N, len(tspan))

print(A2.shape)




# next we try LU decomp


P, L, U = lu(b_matA.toarray())
# Pb = np.dot(P, b)
# y = solve_triangular(L, Pb, lower=True)
# x = solve_triangular(U, y)


def b_lu_rhs(t, w, nu, A, B, C, P, L, U):
    # first get psi
    Pw = np.dot(P, w)
    y = solve_triangular(L, Pw, lower=True)
    psi = solve_triangular(U, y)


    # now put it all together! 
    rhs = (nu*A.dot(w) - (B.dot(psi)) * (C.dot(w)) + (C.dot(psi)) * (B.dot(w)))

    return rhs



# start our timer
start_time = time.time() # Record the start time

# solve! 
sol = solve_ivp(b_lu_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", 
                t_eval=tspan, args=( nu, b_matA, matB, matC, P, L, U))
sol = sol.y.reshape(nx, nx, len(tspan))

# how long did that take?
end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"LU Elapsed Time: {elapsed_time:.2f} seconds")


split = 1
for j, t in enumerate(tspan):
    if (j % split == 0):
        w = sol[:, :, j]
        plt.subplot(3, 3, j//split + 1)
        plt.pcolor(x, y, w, shading='auto')
        plt.title(f'Time: {t}')
        plt.colorbar()

plt.tight_layout()
plt.show()

A3 = sol.reshape(N, len(tspan))

print(A3.shape)

# try BICGSTAB

residuals_bc = []
def bc_callback(residual_norm):
    residuals_bc.append(residual_norm)

def b_bicgstab_rhs(t, w, nu, A, B, C):
    psi, exitcode = bicgstab(A, w, rtol = 1e-4, callback = bc_callback)

    rhs = (nu*A.dot(w) - (B.dot(psi)) * (C.dot(w)) + (C.dot(psi)) * (B.dot(w)))

    return rhs

# start our timer
# start_time = time.time() # Record the start time

# # solve! 
# sol = solve_ivp(b_bicgstab_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", 
#                 t_eval=tspan, args=( nu, b_matA, matB, matC))
# sol = sol.y.reshape(nx, nx, len(tspan))

# # how long did that take?
# end_time = time.time() # Record the end time
# elapsed_time = end_time - start_time
# print(f"BICGSTAB Elapsed Time: {elapsed_time:.2f} seconds")



# split = 1
# for j, t in enumerate(tspan):
#     if (j % split == 0):
#         w = sol[:, :, j]
#         plt.subplot(3, 3, j//split + 1)
#         plt.pcolor(x, y, w, shading='auto')
#         plt.title(f'Time: {t}')
#         plt.colorbar()

# plt.tight_layout()
# plt.show()




## try the gmres or whatever 

residuals_gm = []
def gm_callback(residual_norm):
    residuals_gm.append(residual_norm)


def b_gmres_rhs(t, w, nu, A, B, C):
    psi, exitcode = gmres(A, w, atol = 1e-4, callback = gm_callback)

    rhs = (nu*A.dot(w) - (B.dot(psi)) * (C.dot(w)) + (C.dot(psi)) * (B.dot(w)))

    return rhs

# start our timer
# start_time = time.time() # Record the start time

# # solve! 
# sol = solve_ivp(b_gmres_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", 
#                 t_eval=tspan, args=( nu, b_matA, matB, matC))
# sol = sol.y.reshape(nx, nx, len(tspan))

# # how long did that take?
# end_time = time.time() # Record the end time
# elapsed_time = end_time - start_time
# print(f"GMRES Elapsed Time: {elapsed_time:.2f} seconds")



# split = 1
# for j, t in enumerate(tspan):
#     if (j % split == 0):
#         w = sol[:, :, j]
#         plt.subplot(3, 3, j//split + 1)
#         plt.pcolor(x, y, w, shading='auto')
#         plt.title(f'Time: {t}')
#         plt.colorbar()

# plt.tight_layout()
# plt.show()




### part c


# now we can get a bit creative with it

tspan = np.arange(0, 12, 1)
print(tspan)
w0 = -1 * np.exp(-(X+1)**2 - .05 * Y**2) - 1 * np.exp(-(X-1)**2 - .05 * Y**2) 

# Solve the ODE and plot the results

# start our timer
start_time = time.time() # Record the start time

# solve! 
sol = solve_ivp(a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", 
                t_eval=tspan, args=(nx, ny, N, K, nu, matA, matB, matC))
sol = sol.y.reshape(nx, nx, len(tspan))

# how long did that take?
end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"Part C Elapsed Time: {elapsed_time:.2f} seconds")


split = 1
for j, t in enumerate(tspan):
    if (j % split == 0):
        w = sol[:, :, j]
        plt.subplot(4, 3, j//split + 1)
        plt.pcolor(x, y, w, shading='auto')
        plt.title(f'Time: {t}')
        plt.colorbar()

plt.tight_layout()
plt.show()




### part d 

# animations! 
def animate_plot(sol, x, y, tspan, save=False, filename='animation.mp4', interval = 100):
    """
    Creates an animation of the solution over time.

    Parameters:
    - sol: numpy array of shape (n, n, num_times)
    - x, y: spatial coordinates
    - tspan: array of time points
    - rows, cols: layout of subplots (not used here but kept for compatibility)
    - save: bool, whether to save the animation
    - filename: string, filename for saving the animation
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.pcolor(x, y, sol[:, :, 0], shading='auto', cmap='jet')
    ax.set_title(f'Time: {tspan[0]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(cax, ax=ax, label='Vorticity')

    def update(frame):
        ax.clear()
        cax = ax.pcolor(x, y, sol[:, :, frame], shading='auto', cmap='jet')
        ax.set_title(f'Time: {tspan[frame]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return cax,

    ani = animation.FuncAnimation(
        fig, update, frames=range(sol.shape[2]), blit=False, interval=interval, repeat=False
    )

    if save:
        # To save as MP4, ensure ffmpeg is installed
        ani.save(filename, writer='ffmpeg')
        print(f'Animation saved as {filename}')
    else:
        plt.show()

animate_plot(sol, x, y, tspan)