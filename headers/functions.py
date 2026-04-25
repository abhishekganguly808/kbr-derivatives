import numpy as np
# #Functions

def Camel1D(X):
    dim = 1
    sigma = 0.2
    a = 1/3.
    b = 2/3.

    # Compute squared terms for exponentials
    a1 = (X - a)**2
    b1 = (X - b)**2

    # Gaussian function components
    norm_factor = (sigma * np.pi**0.5)**dim
    exp_a = np.exp(-a1 / sigma**2)
    exp_b = np.exp(-b1 / sigma**2)

    # Function F
    F = 0.5 * (exp_a + exp_b) / norm_factor

    # First derivative dF/dX
    dFdX = 0.5 * (-2 * (X - a) / sigma**2 * exp_a - 2 * (X - b) / sigma**2 * exp_b) / norm_factor

    # Second derivative d2F/dX2
    d2FdX2 = 0.5 * (
        (4 * (X - a)**2 / sigma**4 - 2 / sigma**2) * exp_a +
        (4 * (X - b)**2 / sigma**4 - 2 / sigma**2) * exp_b
    ) / norm_factor

    return F, dFdX, d2FdX2

    

def Weierstrass(X,A=0.75,B=5):
    n = int(input("Enter number of terms of Weierstrass: ")) 
    F =  np.zeros(len(X))
    dFdX = np.zeros(len(X))
    d2FdX2 = np.zeros(len(X))

    for i in range(n):
        F = F + (A**i)*np.cos(B**i * np.pi * X)
        dFdX -= (A**i) * B**i * np.pi * np.sin(B**i * np.pi * X)
        d2FdX2 -= (A**i) * (B**i * np.pi)**2 * np.cos(B**i * np.pi * X)
    
    return F, dFdX, d2FdX2, n


def Piecewise(X):
    F = np.zeros_like(X)
    dFdX = np.zeros_like(X)
    d2FdX2 = np.zeros_like(X)

    # Region 1: [0, 0.2]
    mask1 = (X >= 0) & (X <= 0.2)
    F[mask1] = np.exp(-0.2509 * X[mask1])
    dFdX[mask1] = -0.2509 * np.exp(-0.2509 * X[mask1])
    d2FdX2[mask1] = (0.2509**2) * np.exp(-0.2509 * X[mask1])

    # Region 2: (0.2, 0.4]
    mask2 = (X > 0.2) & (X <= 0.4)
    F[mask2] = np.sin(3 * np.pi * X[mask2])
    dFdX[mask2] = 3 * np.pi * np.cos(3 * np.pi * X[mask2])
    d2FdX2[mask2] = -9 * np.pi**2 * np.sin(3 * np.pi * X[mask2])

    # Region 3: (0.4, 0.6]
    mask3 = (X > 0.4) & (X <= 0.6)
    F[mask3] = 677.3 * X[mask3]**3 - 786.6 * X[mask3]**2 + 296.6 * X[mask3] - 36.7
    dFdX[mask3] = 3 * 677.3 * X[mask3]**2 - 2 * 786.6 * X[mask3] + 296.6
    d2FdX2[mask3] = 6 * 677.3 * X[mask3] - 2 * 786.6

    # Region 4: (0.6, 1.5]
    mask4 = (X > 0.6) & (X <= 1.5)
    F[mask4] = 10 * np.exp(-2.3 * X[mask4]**2)
    dFdX[mask4] = -2 * 2.3 * 10 * X[mask4] * np.exp(-2.3 * X[mask4]**2)
    d2FdX2[mask4] = 10 * np.exp(-2.3 * X[mask4]**2) * (-2.3 + 4 * 2.3**2 * X[mask4]**2)

    # Region 5: (1.5, 1.8]
    mask5 = (X > 1.5) & (X <= 1.8)
    F[mask5] = 3.0
    dFdX[mask5] = 0.0
    d2FdX2[mask5] = 0.0

    # Region 6: X > 1.8
    mask6 = (X > 1.8)
    F[mask6] = 0.0
    dFdX[mask6] = 0.0
    d2FdX2[mask6] = 0.0

    return F, dFdX, d2FdX2


def generic_initial_condition(x_mapped):
    """
    Rescaled initial condition to fit in [0.1, 0.8] from original [-1, 1] domain.
    Returns:
        F : function value
        dFdx, d2Fdx2 : nan (not computed)
    """

    # Convert x from [0.1, 0.8] domain to [-1, 1]
    x = 2 * (x_mapped - 0.1) / 0.7 - 1.0

    # Constants from paper
    z = -0.7
    delta = 0.005
    beta = (np.log(2)**2) / (36 * delta**2)
    a = 0.5
    alpha = 10

    # Helper functions
    def G(x, beta, z):
        return np.exp(-beta * (x - z)**2)

    def F_ellipse(x, alpha, a):
        val = 1 - alpha**2 * (x - a)**2
        return np.sqrt(np.maximum(val, 0))

    x = np.asarray(x)
    F = np.zeros_like(x)

    # Regions in original [-1, 1] domain
    mask_gaussian = (x >= -0.8) & (x <= -0.6)
    mask_triangle = (x >= -0.4) & (x <= -0.2)
    mask_square   = (x >= 0.0)  & (x <= 0.2)
    mask_semiellipse = (x >= 0.4) & (x <= 0.6)

    # Gaussian region
    F[mask_gaussian] = 0.5 * (
        G(x[mask_gaussian], beta, z - delta)
        + 4 * G(x[mask_gaussian], beta, z)
        + G(x[mask_gaussian], beta, z + delta)
    )

    # Triangle
    F[mask_triangle] = 1.0

    # Square wave (triangle shape)
    F[mask_square] = 1.0 - 10 * np.abs(x[mask_square] - 0.1)

    # Semi-ellipse
    F[mask_semiellipse] = 0.5 * (
        F_ellipse(x[mask_semiellipse], alpha, a - delta)
        + 4 * F_ellipse(x[mask_semiellipse], alpha, a)
        + F_ellipse(x[mask_semiellipse], alpha, a + delta)
    )

    # Derivatives not defined (discontinuities)
    dFdx = np.full_like(x, np.nan)
    d2Fdx2 = np.full_like(x, np.nan)

    return F, dFdx, d2Fdx2


def sine(X,k=1.0):
    F = np.sin(np.pi * X * k ) 
    dFdX = k * np.pi * np.cos(np.pi * X * k)
    d2FdX2 = - (k**2)*(np.pi**2) * np.sin(np.pi * X * k)
    return F, dFdX, d2FdX2

def gaussian(X, x0=0.5, alpha=100):
    F = np.exp(-alpha * (X - x0) ** 2)  
    dFdX = -2 * alpha * (X - x0) * F  # First derivative
    d2FdX2 = (4 * alpha**2 * (X - x0)**2 - 2 * alpha) * F  # Second derivative    
    return F, dFdX, d2FdX2


def Polynomial(X,a=0,b=0,c=0,d=0,e=0):
    F= a+ b*X + c*X**2 + d*X**3 + e*X**4
    dFdX = 2*c*X + b + 3*d*X**2 + 4*e*X**3
    d2FdX2 = 2*c + 6*d*X + 12*e*X**2
    return F, dFdX, d2FdX2



def AddNoise(F,type, n_scale):
    np.random.seed(45)
    # Generate Gaussian noise with mean 0 and standard deviation 0.05
    if type == 'Gaussian':
        G = np.random.normal(0, 1/3, np.shape(F))
        # Add Gaussian noise to F
        F = F*(1 + n_scale*G)
    elif type == 'Uniform':
        U  = (2.0*np.random.random(np.shape(F))-1.0)
        F = F*(1+U*n_scale)
    return F


def Normalize(F,scale):
    F = (F)/scale  #Scales the data between -1.0 and 1.0
    return F

    
def sawtooth_wave(X, period=0.25):
    return 2 * ((X / period) % 1) - 1

def step_function(X, threshold=0.5):
    F = np.where(X < threshold, 0.0, 1.0)
    dFdX = np.zeros_like(X)  # Derivative of step function is zero everywhere except at threshold (discontinuous)
    d2FdX2 = np.zeros_like(X)  # Second derivative is also zero everywhere
    return F, dFdX, d2FdX2


def rectangular_pulse(X):
    return np.where((X % 1) < 0.5, 1.0, -1.0)

def triangle_wave(X, period=1):
    return 2 * np.abs((X / period) % 1 - 0.5) - 1

def hat_function(X):
    F = np.where((X > 0.4) & (X < 0.6), 1.0, 0.0)
    dFdX = np.where((X == 0.4) | (X == 0.6), np.inf, 0.0)  # Infinite slope at discontinuities  
    d2FdX2 = np.where((X == 0.4) | (X == 0.6), np.inf, 0.0)  # Second derivative is also singular    
    return F, dFdX, d2FdX2


def riemann_function(X, left_val = 1.0, right_val = 0.0, discont = 0.5):
    F = np.where((X < discont), left_val, right_val)
    dFdX = np.where((X == discont),  np.nan, 0.0)  # Infinite slope at discontinuities  
    d2FdX2 = np.where((X == discont) , np.nan, 0.0)  # Second derivative is also singular    
    return F, dFdX, d2FdX2





def Rastrigin1D(x , A=1):
    k = 5      
    F = x**2 - A * np.cos(k * np.pi * x) + A
    dFdX = 2 * x + A * k * np.pi * np.sin(k * np.pi * x)
    d2FdX2 = 2 + A * (k * k * np.pi**2) * np.cos(k * np.pi * x)
    return F, dFdX, d2FdX2

