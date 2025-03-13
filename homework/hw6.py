import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import kron


# Define parameters
tspan = np.arange(0, 4.5, .5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny
beta, D1, D2 = 1,.1,.1


#### fast fourier transform bit!

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6 # reset initial value! 
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6 # for both x and y! 
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# initialize u and v for our particular problem 
m=1 # number of spirals
u0=np.tanh(np.sqrt(X**2+Y**2))*np.cos(m*np.angle(X+1j*Y)-(np.sqrt(X**2+Y**2)))
v0=np.tanh(np.sqrt(X**2+Y**2))*np.sin(m*np.angle(X+1j*Y)-(np.sqrt(X**2+Y**2)))

w0t = np.hstack([np.fft.fft2(u0).flatten(), np.fft.fft2(v0).flatten()])

# shoot function working within fourier domain
def fft_rhs(t, w0, nx, ny, N, K):

    # w0 is stacked u and v
    ut = w0[:N].reshape((nx,ny))
    vt = w0[N:].reshape((nx,ny))
    u = np.real(np.fft.ifft2(ut))
    v = np.real(np.fft.ifft2(vt))

    # A2 and reaction terms 
    A2 = u**2 + v**2
    lambdaA2 = 1 - A2
    omegaA2 = - beta*A2 

    # fourier transform according to the differential equations
    u2t = np.fft.fft2(lambdaA2*u) - np.fft.fft2(omegaA2*v) - D1*K*ut
    v2t = np.fft.fft2(omegaA2*u) + np.fft.fft2(lambdaA2*v) - D2*K*vt

    return np.hstack([u2t.flatten(), v2t.flatten()])

# solve_ivp and plot the results
sol = solve_ivp(fft_rhs, [tspan[0], tspan[-1]], w0t, method="RK45", 
                t_eval=tspan, args=(nx, ny, N, K))

A1 = np.real(sol.y)
#print(A1)



# plot our beautiful spirals
for j, t in enumerate(tspan):
    w = np.real(np.fft.ifft2(sol.y.T[j,:N].reshape((nx, ny))))
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w, shading='interp')
    plt.title(f'Time: {t}')
    plt.colorbar()

plt.tight_layout()
plt.show()



## chebychevvv

# create chebychev matrix
# this returns both x domain and the matrix 
def cheb(N):
    if N==0:
        D = 0
        x = 1
    else:
        n = np.arange(0,N+1)
        x = np.cos(np.pi*n/N).reshape(N+1,1)
        c = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
        X = np.tile(x,(1,N+1))
        dX = X - X.T
        D = np.dot(c,1./c.T)/(dX+np.eye(N+1))
        D -= np.diag(np.sum(D.T, axis=0))
    return D, x.reshape(N+1)

# initial conditions! collapse nx and ny into just n for simplicity
n = 31
N = 30
L = 20

# initialize our domain and derivative matrix 
D, x = cheb(N)
y = x 
X, Y = np.meshgrid(x, y)

# rescale to -1 to 1 for cheb function
X = X*(L/2)
Y = Y*(L/2)

# update boundary conditions: no flux
D[0, :] = 0 
D[N, :] = 0 

# recreate our initial conditions 
theta = np.angle(X + 1j * Y)
r = np.sqrt(X**2 + Y**2)
u0 = (np.tanh(r) * np.cos(m * theta - r))
v0 = (np.tanh(r) * np.sin(m * theta - r))
w0 = np.concatenate([u0.flatten(), v0.flatten()])

# set up dual second derivative matrix 
D_xx = (np.dot(D,D))/(L/2)**2
I = np.eye(len(D))  
laplacian = np.kron(I, D_xx) + np.kron(D_xx, I) 

# cheb right hand side
def cheb_rhs(t, UV):
    u = UV[:n*n].reshape((n, n))
    v = UV[n*n:].reshape((n, n))

    u_flat = u.flatten()
    v_flat = v.flatten()
    A2 = u**2 + v**2
    lambda_A = 1-A2
    omega_A = -beta*A2

    u_t = (D1*np.dot(laplacian, u_flat)).reshape((n, n)) + lambda_A*u - omega_A*v
    v_t = (D2*np.dot(laplacian, v_flat)).reshape((n, n)) + omega_A*u + lambda_A*v
    
    return np.concatenate([u_t.flatten(), v_t.flatten()])

sol = solve_ivp(cheb_rhs, [tspan[0],tspan[-1]], w0, t_eval=tspan, method='RK45')
A2 = sol.y


n_timesteps = len(tspan)
fig, axes = plt.subplots(2, n_timesteps // 2, figsize=(15, 6))
for i, ax in enumerate(axes.flatten()):
    u = sol.y[:n * n, i].reshape((n, n))
    im = ax.imshow(u, extent=(-L/2, L/2, -L/2, L/2), origin='lower', cmap='viridis')
    ax.set_title(f't = {tspan[i]:.1f}') 
    fig.colorbar(im, ax=ax, orientation='vertical')
plt.tight_layout()
plt.show()

print(A2)