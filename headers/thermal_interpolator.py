import numpy as np
from numba import njit
from scipy.interpolate import Rbf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


#norms
def L1(A,B):
    A, B = np.asarray(A), np.asarray(B)
    if A.shape != B.shape:
        B = B.reshape(A.shape)
    return np.mean(abs(A-B))

def L2(A, B):
    A, B = np.asarray(A), np.asarray(B)   
    if A.shape != B.shape:
        B = B.reshape(A.shape)
    
    nan_mask = ~np.isnan(A) & ~np.isnan(B)   
    if not np.all(nan_mask):
        nan_indices = np.where(~nan_mask)[0]
        print(f"NaN detected at indices: {nan_indices}. These elements will be excluded from L2 computation.")
    
    # A = A[nan_mask]
    # B = B[nan_mask]
    
    return np.sqrt(np.mean((A - B) ** 2))

def Linf(A,B):
    A, B = np.asarray(A), np.asarray(B)   
    if A.shape != B.shape:
        B = B.reshape(A.shape)
    return np.max(abs(A-B))






def scaled_sigma_knn(points, k=5, c=1.0):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points[:, None])
    distances, _ = nbrs.kneighbors(points[:, None])
    mean_knn_dist = np.mean(distances[:, 1:])  # Exclude self-distance
    return c * mean_knn_dist



def rbf_get_theta(data, theta0=-1, maX_iter=30, alpha=0.5, threshold_input=None, init_mode='knn'):
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    mean_knn_dist = np.mean(distances[:, 1:])  # exclude self-distance
    threshold = ( threshold_input if threshold_input is not None 
                        else (0.01 * mean_knn_dist ** 2))
    Npoints, dim = data.shape
    a = 1    
    # array for keeping track of theta values
    theta_lst = np.zeros(maX_iter)
    
    # constants
    scale = dim/Npoints
    R2    = np.sum( (data - np.mean(data, axis=0))**2, axis=1 )

    # Initial theta
    if theta0 < 0:
        if init_mode == 'variance':
            theta0 = (1. / (dim*Npoints))*np.sum( (data - np.mean(data, axis=0))**2  )    
        elif init_mode == 'knn':
            theta0 = 10*mean_knn_dist**2
        else:
            raise ValueError("init_mode must be 'variance' or 'knn'")
    
    theta_lst[0] = theta_old = theta0

    for idx in range(1, maX_iter):                

        coeff = 1.0 / (2.0 * theta_old)

        num  = np.sum(R2 * ( np.exp(-R2*coeff))) 
        den  = np.sum(     ( np.exp(-R2*coeff))) 
        frac = scale*(num / den)

        # get new value of theta
        theta_new = (alpha)*frac + (1.0-alpha)*theta_old

        theta_lst[idx] = theta_new


        # exit if below threshold or Nan
        if theta_new < threshold or np.isnan(theta_new):
            theta_new = theta_old
            theta_lst[idx] = theta_new
            a = 0
            # print(f"triggered! revert theta:{theta_new}")
            break
        
        theta_old      = theta_new
    
    # return non-zero values
    return theta_lst[:idx+a*1] , mean_knn_dist   


def solve_shift_bisection(x_target, x_train, theta_inv_half, tol=1e-12, max_iter=5): # do not change the number of iterations. convergence reuslts may change
    dx_eff = 1.0/np.shape(x_train)[0]
    low = x_target-dx_eff
    high = x_target+dx_eff
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_vec = np.array([mid])  # ensure it's array-like

        pred = predict_linear_numba(theta_inv_half, x_train, mid_vec)[0]  # return scalar

        error = pred - x_target
        if abs(error) < tol:
            return mid_vec

        if error > 0:
            high = mid
        else:
            low = mid

    return np.array([mid])  # ensure shape consistency

def lagrange_multiplier_1D(theta, x_test, x_train, number_of_iterations):
    NPoints = x_test.shape[0]
    LM      = np.zeros(NPoints)  
    x_test_tilde    = x_test.copy()
    theta_inv_half = 0.5 / theta 
    for idx in range( NPoints ):
        # print(f"for x_test = {x_test[idx]}")
        err_temp_prev = 10
        distance = (x_test[idx,:] - x_train)
        x_test_tilde[idx] = solve_shift_bisection(x_test[idx], x_train, theta_inv_half)
        for iter in range(number_of_iterations):
            distance_x_test_tilde = (x_test_tilde[idx,:] - x_train)
            distance_x_test_tilde_sqr = distance_x_test_tilde ** 2

            R_sqr_new = distance_x_test_tilde_sqr * (theta_inv_half)

            probability_indiv = np.exp(-R_sqr_new)


            num = np.sum(distance * probability_indiv)
            den = np.sum(distance * distance_x_test_tilde * probability_indiv)
            if den == 0 or np.isnan(den):
                break
            LM[idx] =  theta * num / den
            res_temp = predict_linear_numba(theta_inv_half, x_train, x_test_tilde[idx]+LM[idx], eps=1e-64)
            # print(f"for x_test = {x_test[idx]}, LM is {LM[idx]}, res temp is  {res_temp}")

            err_temp = (res_temp-x_test[idx])
        
            if np.abs(err_temp) < 1e-15:
                    x_test_tilde[idx] += LM[idx]
                    break  # Stop if error is sufficiently small

            elif np.abs(err_temp) < np.abs(err_temp_prev):

                    # Update the solution if error has decreased
                    x_test_tilde[idx] += LM[idx]
                    
                    # Check if the error has sufficiently converged (less than 5% relative change)
                    # if abs(err_temp_prev - err_temp)  <  abs(0.05 *  err_temp_prev):
                        # break  # Exit if convergence threshold met
                    
                    # Update the previous error for the next iteration
                    err_temp_prev = err_temp
                    # print(f"for x_test = {x_test[idx]}, new X is {x_test_tilde[idx]}, error for iter {iter} is {err_temp}")

            else:

                    # print(f"for x_test = {x_test[idx]}, new X is {x_test_tilde[idx]}, error for iter {iter} is {err_temp}")

                    break  # Exit if no improvement in error or increment in error from NR oscillations.
        # print(f"for x_test = {x_test[idx]}, modified is {x_test_tilde[idx]} ")   
        # print(f"------------------------------------------------------")                
             
    return x_test_tilde




################ Does the summation [==SUM(P(X_i,x)*Y_i)]
@njit
def predict_numba(theta, X_train, Y_train, X_test, eps=1e-64, verbose = 0):
                
    Npoints = X_test.shape[0]
    interpolated_value     = np.zeros(Npoints)
    if verbose == 0:
        for idx in range( Npoints ):
        
            distance_sqr    = np.sum( (X_test[idx,:] - X_train[:,:])**2  , axis=1)
            probability_i      = ( np.exp( -distance_sqr / (2.0*theta)) )
            probability_sum    = np.sum(probability_i)


            interpolated_value[idx] = np.sum(Y_train*probability_i) / (probability_sum + eps)
    # if verbose == 1:
    #     for idx in range(Npoints):
    #         if idx == 0 or idx == Npoints-1:
    #             print("for i =", idx)

    #             distance_sqr   = np.sum((X_test[idx, :] - X_train[:, :])**2, axis=1)
    #             probability_i  = np.exp(-distance_sqr / (2.0 * theta))
    #             probability_sum = np.sum(probability_i)

    #             interpolated_value[idx] = np.sum(Y_train * probability_i) / (probability_sum + eps)

    #             # Element-wise print (Numba-compatible)
    #             print("-"*50)

    #             for j in range(len(Y_train)):
    #                 print("(X[%d] - X_train[%d])^2 = (%.2e - %.2e)^ = %.4e" %
    #                     (j, j, X_test[idx], X_train[j], (X_test[idx] - X_train[j])**2))
    #                 # print("Y[%d] * r^2[%d] = %.6e * %.6e = %.6e" %
    #                 #     (j, j, Y_train[j], distance_sqr[j], Y_train[j] * probability_i[j]))
    return interpolated_value

################ Does the summation [==SUM(P(X_i,x)*Y_i)]
@njit
def predict_derivative_numba(theta, X_train, Y_train, X_test, eps=1e-64):
                
    Npoints = X_test.shape[0]
    interpolated_derivative_value     = np.zeros(Npoints)
    theta_inv = 1.0/theta
    for idx in range( Npoints ):
    
        distance_sqr    = np.sum( (X_test[idx,:] - X_train[:,:])**2  , axis=1)
        distance        = np.sum( (X_test[idx,:] - X_train[:,:])  , axis=1)
        probability_i      = ( np.exp( -0.5*distance_sqr*theta_inv) )
        probability_sum    = np.sum(probability_i)


        interpolated_derivative_value[idx] = np.sum(Y_train*(-distance*theta_inv)*probability_i*(probability_sum-probability_i)) / (probability_sum + eps)**2
         
    return interpolated_derivative_value


################ Calculates the Lagrange Multiplier using an iterative Newton Raphson for N-Dimensions [x = SUM(P(x_i,x) *x_i)]
# @njit
def lagrange_multiplier_NDim_numba(theta, x_test, x_train, number_of_iterations):
    NPoints = x_test.shape[0]
    Dim =   x_test.shape[1]
    LM      = np.zeros((NPoints,Dim))
    x_tilde = x_test.copy()
    J = np.zeros((Dim,Dim))
    F = np.zeros((Dim,1))
    theta_inv_half = 0.5 / theta 
    for idx in range(NPoints):
        err_temp_prev = 10
        distance = (x_test[idx] - x_train)
        for iterations in range(number_of_iterations):
            distance_beta = (x_tilde[idx] - x_train)
            R_sqr_new = (np.sum(distance_beta ** 2,axis = 1)) *theta_inv_half   


            probability_indiv =  np.exp(-R_sqr_new)
            for row in range(Dim):
                temp = distance[:,row] * probability_indiv
                F[row][0] = np.sum(temp)
                for col in range(Dim):
                    J[row][col] =  np.sum(temp * distance_beta[:,col]) 
          
		# This is to avoid the try/expect part when matrix is singular
            # J += np.eye(Dim) * 1e-32
            try:
                LM[idx] = theta * np.transpose(np.linalg.solve(J,F))
            except Exception:
                break
            # Solve the linear system
            # LM[idx] = theta * np.transpose(np.linalg.solve(J, F))

            res_temp = predict_linear_numba(theta_inv_half  , x_train, x_tilde[idx]+LM[idx], eps=1e-64)
            err_temp = np.sqrt(np.mean((res_temp - x_test[idx])**2))

            ############# Case Handling
            if np.abs(err_temp) < np.abs(err_temp_prev):
                # Update the solution if error has decreased
                x_tilde[idx] += LM[idx]
                
                # Check if the error has sufficiently converged (less than 5% relative change)
                if abs(err_temp_prev - err_temp)  <  abs(0.05 *  err_temp_prev):
                    break  # Exit if convergence threshold met
                
                # Update the previous error for the next iteration
                err_temp_prev = err_temp
            else:
                break  # Exit if no improvement in error or increment in error from NR oscillations.

    return x_tilde


# @njit
def predict_linear_numba(theta_inv_half, X_train, X_test, eps=1e-64):
    Dim = len(X_test)
    interpolated_value_linear     = np.zeros(Dim)
    for coord in range(Dim):
        y_train = (X_train)[:,coord]
        distance_sqr    = np.sum( (X_test - X_train)**2  , axis=1)
        probability_i      = ( np.exp( -distance_sqr * (theta_inv_half)) )  
        probability_sum    = np.sum(probability_i)

        interpolated_value_linear[coord] = np.sum(y_train*probability_i) / (probability_sum + eps)
            
    return interpolated_value_linear

################ Does the summation [==SUM(P(X_i,x)*X_i^2)]
@njit
def predict_sec_moment_numba(theta, X_train, X_test, eps=1e-64):
                
    Npoints = X_test.shape[0]
    interpolated_value     = np.zeros(Npoints)

    for idx in range( Npoints ):
    
        distance_sqr    = np.sum( (X_test[idx,:] - X_train[:,:])**2  , axis=1)
        probability_i      = ( np.exp( -distance_sqr / (2.0*theta)) )
        probability_sum    = np.sum(probability_i)


        interpolated_value[idx] = np.sum((X_train.ravel()**2)*probability_i) / (probability_sum + eps)
         
    return interpolated_value




# @njit
def predict_quadratic_numba(theta_inv_half, X_train, X_test, eps=1e-64):
    Dim = len(X_test)
    interpolated_value_quadratic     = np.zeros(Dim)
    for coord in range(Dim):
        y_train = ((X_train)[:,coord])**2   
        distance_sqr    = np.sum( (X_test - X_train)**2  , axis=1)
        probability_i      = ( np.exp( -distance_sqr * (theta_inv_half)) )  
        probability_sum    = np.sum(probability_i)

        interpolated_value_quadratic[coord] = np.sum(y_train*probability_i) / (probability_sum + eps)
            
    return interpolated_value_quadratic


def interpolator(dim, validation_data, train_data=None, test_data=None,  noise_token=0, corrections_key=1 , theta_o = None, name = None):
    # Set some parameters
    max_iter = 5
    r_seed = 25
    np.random.seed(r_seed)
    theta_optimum   = 0.0 
    test_size_fraction = 0.2
    
    if validation_data is not None and train_data is None and test_data is None:
            train_data, test_data = train_test_split(validation_data, test_size=test_size_fraction, shuffle=True)


    # Extract dimensions
    N_train = train_data.shape[0]
    N_test = test_data.shape[0]

    X_train = train_data[:,:dim]
    X_test  = test_data[:,:dim]
    x_validation = validation_data[:,:dim]
    Y_train_base = train_data[:,dim]
    Y_test_base  = test_data[:,dim]

    # set token to 1 to work with noisy data:
    
    if noise_token == 1:
        Y_train = train_data[:,dim+1]
        Y_test  = test_data[:,dim+1]  
  
    else: 
        Y_train = Y_train_base
        Y_test  = Y_test_base


    print(f"Size of training data {N_test+N_train}")
    print(f"Size of deployment data {validation_data.shape[0]}")
    print(f"Noise token is {noise_token}, Correction token is {corrections_key}")


    theta_lst, _ = rbf_get_theta(X_train, maX_iter=15, threshold_input=None, init_mode='knn')

    print(f"theta: {theta_lst}")
    mseLst = []
    if dim != 1:    
        if corrections_key == 1:
            if theta_o == None:
                with open(f'./results/{dim}D/thetalist_{dim}D_corrected.txt', 'w') as f:
                    f.write(f'# theta   test_rmse   base_rmse\n')
                    for theta in theta_lst:
                        del_X_train = lagrange_multiplier_NDim_numba(theta, X_train, X_train, max_iter)
                        del_X_test = lagrange_multiplier_NDim_numba(theta, X_test, X_train, max_iter)
                        Y_train_predict = predict_numba(theta, X_train, Y_train, del_X_train, eps=1e-64)
                        res = predict_numba(theta, X_train, 2 * Y_train - Y_train_predict, del_X_test, eps=1e-64)
                        RMSE = L2(res, Y_test)
                        RMSE_base = L2(res, Y_test_base)

                        f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                        mseLst.append(RMSE)

                    idx = np.argmin(mseLst)
                    theta_optimum = theta_lst[idx]
            else:
                theta_optimum = theta_o

            print(f"Selected theta is {theta_optimum}, with test rmse {mseLst[idx]}")
            del_X_train = lagrange_multiplier_NDim_numba(theta_optimum, X_train, X_train, max_iter)
            # Self-correction step
            Y_train_predict = predict_numba(theta_optimum, X_train, Y_train, del_X_train, eps=1e-64)
            del_X_validation = lagrange_multiplier_NDim_numba(theta_optimum, x_validation, X_train, max_iter)
            Y_validation_predicted = predict_numba(theta_optimum, X_train, 2 * Y_train - Y_train_predict, del_X_validation, eps=1e-64)

        elif corrections_key == 0:
            if theta_o == None:

                with open(f'./results/{dim}D/thetalist_{dim}D_uncorrected.txt', 'w') as f:
                    f.write(f'# theta   test_rmse   base_rmse\n')
                    for theta in theta_lst:
                        res = predict_numba(theta, X_train, Y_train, X_test, eps=1e-64)
                        RMSE = L2(res, Y_test)
                        RMSE_base = L2(res, Y_test_base)

                        f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                        mseLst.append(RMSE)
                    idx = np.argmin(mseLst)
                    theta_optimum = theta_lst[idx]
            else:
                theta_optimum = theta_o           
            print(f"Selected theta is {theta_optimum}, with test rmse {mseLst[idx]}")
            Y_validation_predicted = predict_numba(theta_optimum, X_train, Y_train, x_validation, eps=1e-64)

        with open(f'./results/{dim}D/predictions_{dim}D.txt', 'w') as f:
            for idx in range(len(Y_validation_predicted)):
                f.write(f'{Y_validation_predicted[idx]}\n')  
        # print(f"Complete!")

    else:
        if corrections_key == 1:
            if theta_o == None:
                with open(f'./datafiles/thetalist_1D_corrected_v2_{name}.txt', 'w') as f:
                    f.write(f'# theta   test_rmse   base_rmse\n')
                    for theta in theta_lst:
                        del_X_train = lagrange_multiplier_1D(theta, X_train, X_train, max_iter)
                        del_X_test = lagrange_multiplier_1D(theta, X_test, X_train, max_iter)
                        Y_train_predict = predict_numba(theta, X_train, Y_train, del_X_train, eps=1e-64)
                        theta_train = predict_sec_moment_numba(theta, X_train, del_X_train, eps = 1e-64) - (np.ravel(X_train**2))
                        theta_test = predict_sec_moment_numba(theta, X_train, del_X_test, eps = 1e-64) - (np.ravel(X_test**2))
                        c = (Y_train - Y_train_predict) / (theta_train)
                        c_clean = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                        res = predict_numba(theta, X_train, Y_train , del_X_test, eps=1e-64) + theta_test * predict_numba(theta, X_train, c_clean, del_X_test, eps=1e-64)
                        # res = predict_numba(theta, X_train, 2 * Y_train - Y_train_predict, del_X_test, eps=1e-64)
                        RMSE = L2(res, Y_test) 
                        RMSE_base = L2(res, Y_test_base)
                        f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                        mseLst.append(RMSE)

                    idx = np.argmin(mseLst)
                    theta_optimum = theta_lst[idx]
            else:
                 theta_optimum = theta_o
            
            
            print(f"Selected theta is {theta_optimum}, with test rmse {mseLst[idx]}")

            del_X_train =  lagrange_multiplier_1D(theta_optimum, X_train, X_train, max_iter)
            Y_train_predict = predict_numba(theta_optimum, X_train, Y_train, del_X_train, eps=1e-64)
                
            
            theta_train = predict_sec_moment_numba(theta_optimum, X_train, del_X_train, eps=1e-64) - (np.ravel(X_train**2))
            c = (Y_train - Y_train_predict) / (theta_train)
            c_clean = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            del_x_validation =   lagrange_multiplier_1D(theta_optimum, x_validation, X_train, max_iter)
            theta_validation = predict_sec_moment_numba(theta_optimum, X_train, del_x_validation, eps=1e-64) - (np.ravel(x_validation**2))
            Y_validation_predicted = predict_numba(theta_optimum, X_train, Y_train, del_x_validation, eps=1e-64, verbose = 0) +  theta_validation * predict_numba(theta_optimum, X_train, c_clean, del_x_validation, eps=1e-64)
        
        elif corrections_key == 0:
            if theta_o == None:
                with open(f'./results/thetalist_1D_uncorrected.txt', 'w') as f:
                    f.write(f'# theta   test_rmse   base_rmse\n')
                    for theta in theta_lst:
                        res = predict_numba(theta, X_train, Y_train, X_test, eps=0)
                        RMSE = L2(res, Y_test)
                        RMSE_base = L2(res, Y_test_base)

                        f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                        mseLst.append(RMSE)
                    idx = np.argmin(mseLst)
                    theta_optimum = theta_lst[idx]
            else:
                theta_optimum = theta_o
           
            print(f"Selected theta is {theta_optimum}, with test rmse {mseLst[idx]}")
            Y_validation_predicted = predict_numba(theta_optimum, X_train, Y_train, x_validation, eps=1e-64)

        # with open(f'./results/{dim}D/predictions_1D.txt', 'w') as f:
        #     for idx in range(len(Y_validation_predicted)):
        #         f.write(f'{Y_validation_predicted[idx]}\n')  
        # print(f"Complete!")
    return Y_validation_predicted


def interpolatorMLST(dim, validation_data, train_data=None, test_data=None,  noise_token=0, corrections_key=1 , theta_o = None, name = None):
    # Set some parameters
    max_iter = 5
    r_seed = 25
    np.random.seed(r_seed)
    theta_optimum   = 0.0 
    test_size_fraction = 0.2
    
    if validation_data is not None and train_data is None and test_data is None:
            train_data, test_data = train_test_split(validation_data, test_size=test_size_fraction, shuffle=True)


    # Extract dimensions
    N_train = train_data.shape[0]
    N_test = test_data.shape[0]

    X_train = train_data[:,:dim]
    X_test  = test_data[:,:dim]
    x_validation = validation_data[:,:dim]
    Y_train_base = train_data[:,dim]
    Y_test_base  = test_data[:,dim]

    # set token to 1 to work with noisy data:
    
    if noise_token == 1:
        Y_train = train_data[:,dim+1]
        Y_test  = test_data[:,dim+1]  
  
    else: 
        Y_train = Y_train_base
        Y_test  = Y_test_base


    print(f"Size of training data {N_test+N_train}")
    print(f"Size of deployment data {validation_data.shape[0]}")
    print(f"Noise token is {noise_token}, Correction token is {corrections_key}")


    theta_lst, _ = rbf_get_theta(X_train, maX_iter=15, threshold_input=None, init_mode='knn')

    print(f"theta: {theta_lst}")
    mseLst = []
    if dim != 1:    
        if corrections_key == 1:
            if theta_o == None:
                with open(f'./results/thetalist_{dim}D_corrected.txt', 'w') as f:
                    f.write(f'# theta   test_rmse   base_rmse\n')
                    for theta in theta_lst:
                        del_X_train = lagrange_multiplier_NDim_numba(theta, X_train, X_train, max_iter)
                        del_X_test = lagrange_multiplier_NDim_numba(theta, X_test, X_train, max_iter)
                        Y_train_predict = predict_numba(theta, X_train, Y_train, del_X_train, eps=1e-64)
                        res = predict_numba(theta, X_train, 2 * Y_train - Y_train_predict, del_X_test, eps=1e-64)
                        RMSE = L2(res, Y_test)
                        RMSE_base = L2(res, Y_test_base)

                        f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                        mseLst.append(RMSE)

                    idx = np.argmin(mseLst)
                    theta_optimum = theta_lst[idx]
            else:
                theta_optimum = theta_o

            print(f"Selected theta is {theta_optimum}, with test rmse {mseLst[idx]}")
            del_X_train = lagrange_multiplier_NDim_numba(theta_optimum, X_train, X_train, max_iter)
            # Self-correction step
            Y_train_predict = predict_numba(theta_optimum, X_train, Y_train, del_X_train, eps=1e-64)
            del_X_validation = lagrange_multiplier_NDim_numba(theta_optimum, x_validation, X_train, max_iter)
            Y_validation_predicted = predict_numba(theta_optimum, X_train, 2 * Y_train - Y_train_predict, del_X_validation, eps=1e-64)

        elif corrections_key == 0:
            if theta_o == None:

                with open(f'./results/thetalist_{dim}D_uncorrected.txt', 'w') as f:
                    f.write(f'# theta   test_rmse   base_rmse\n')
                    for theta in theta_lst:
                        res = predict_numba(theta, X_train, Y_train, X_test, eps=1e-64)
                        RMSE = L2(res, Y_test)
                        RMSE_base = L2(res, Y_test_base)

                        f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                        mseLst.append(RMSE)
                    idx = np.argmin(mseLst)
                    theta_optimum = theta_lst[idx]
            else:
                theta_optimum = theta_o           
            print(f"Selected theta is {theta_optimum}, with test rmse {mseLst[idx]}")
            Y_validation_predicted = predict_numba(theta_optimum, X_train, Y_train, x_validation, eps=1e-64)

        with open(f'./results/{dim}D/predictions_{dim}D.txt', 'w') as f:
            for idx in range(len(Y_validation_predicted)):
                f.write(f'{Y_validation_predicted[idx]}\n')  
        # print(f"Complete!")

    else:
        if corrections_key == 1:
            if theta_o == None:
                with open(f'./datafiles/thetalist_1D_corrected_{name}.txt', 'w') as f:
                    f.write(f'# theta   test_rmse   base_rmse\n')
                    for theta in theta_lst:
                        del_X_train = lagrange_multiplier_1D(theta, X_train, X_train, max_iter)
                        del_X_test = lagrange_multiplier_1D(theta, X_test, X_train, max_iter)
                        Y_train_predict = predict_numba(theta, X_train, Y_train, del_X_train, eps=1e-64)
                        res = predict_numba(theta, X_train, 2 * Y_train - Y_train_predict, del_X_test, eps=1e-64)
                        RMSE = L2(res, Y_test)
                        RMSE_base = L2(res, Y_test_base)

                        f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                        mseLst.append(RMSE)

                    idx = np.argmin(mseLst)
                    theta_optimum = theta_lst[idx]
            else:
                 theta_optimum = theta_o
            
            
            print(f"Selected theta is {theta_optimum}, with test rmse {mseLst[idx]}")

            del_X_train = lagrange_multiplier_1D(theta_optimum, X_train, X_train, max_iter)
            # Self-correction step
            Y_train_predict = predict_numba(theta_optimum, X_train, Y_train, del_X_train, eps=1e-64)
            del_X_validation = lagrange_multiplier_1D(theta_optimum, x_validation, X_train, max_iter)
            Y_validation_predicted = predict_numba(theta_optimum, X_train, 2 * Y_train - Y_train_predict, del_X_validation, eps=1e-64)

        elif corrections_key == 0:
            if theta_o == None:
                with open(f'./results/thetalist_1D_uncorrected.txt', 'w') as f:
                    f.write(f'# theta   test_rmse   base_rmse\n')
                    for theta in theta_lst:
                        res = predict_numba(theta, X_train, Y_train, X_test, eps=0)
                        RMSE = L2(res, Y_test)
                        RMSE_base = L2(res, Y_test_base)

                        f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                        mseLst.append(RMSE)
                    idx = np.argmin(mseLst)
                    theta_optimum = theta_lst[idx]
            else:
                theta_optimum = theta_o
           
            print(f"Selected theta is {theta_optimum}, with test rmse {mseLst[idx]}")
            Y_validation_predicted = predict_numba(theta_optimum, X_train, Y_train, x_validation, eps=1e-64)

        # with open(f'./results/{dim}D/predictions_1D.txt', 'w') as f:
        #     for idx in range(len(Y_validation_predicted)):
        #         f.write(f'{Y_validation_predicted[idx]}\n')  
        # print(f"Complete!")
    return Y_validation_predicted







def derivatives_1D_explicit(validation_data, noise_token, training_data=None, theta_o=None, scale = None):
    print("\nrunning explicit")
    max_iter = 30
    dim = 1
    test_size_fraction = 0.1
    np.random.seed(25)
    if scale is not None:
        s = scale
    else:
        s = (np.loadtxt('./temp/temp_scale.txt'))[0]
    # Split data: if no separate training_data is provided, use validation_data.
    if  training_data is not None:
        data = training_data
        data[:, 1:] *= scale
        validation_data[:,1:] *= s
    else:
        data = validation_data 
        data[:, 1:] *= s


    while True:
        train_data, test_data = train_test_split(data, test_size=test_size_fraction, shuffle=True)
        X_test = test_data[:, :dim]
        
        if np.any(X_test == np.max(data[:,0])) or np.any(X_test == np.min(data[:,0])):
            continue  # Retry
        else:
            break  # Success



    N = train_data.shape[0] + test_data.shape[0]
    X_train = train_data[:, :dim]
    X_test = test_data[:, :dim]
    x_validation = validation_data[:, :dim]
    Y_train_base = train_data[:, dim]
    Y_test_base = test_data[:, dim]
    Y_validation_base = validation_data[:, dim]
    if noise_token == 1:
        Y_train = train_data[:, dim+1]
        Y_test = test_data[:, dim+1]
        Y_validation = validation_data[:, dim+1]
    else:
        Y_train = Y_train_base
        Y_test = Y_test_base
        Y_validation = Y_validation_base

    # Obtain candidate theta values.
    theta_lst, effective_dx = rbf_get_theta(X_train, maX_iter=10, threshold_input=None, init_mode='knn')
    # print(f"theta: {theta_lst}")

    # Theta optimization.
    mseLst = []
    if theta_o is None:
        with open('./results/thetalist_1D.txt', 'w') as f:
            f.write('# theta   test_rmse   base_rmse\n')
            for theta in theta_lst:
                del_X_train = lagrange_multiplier_1D(theta, X_train, X_train, max_iter)
                del_X_test = lagrange_multiplier_1D(theta, X_test, X_train, max_iter)
                Y_train_predict = predict_numba(theta, X_train, Y_train, del_X_train, eps=1e-64)
                theta_train = predict_sec_moment_numba(theta, X_train, del_X_train, eps = 1e-64) - (np.ravel(X_train**2))
                theta_test = predict_sec_moment_numba(theta, X_train, del_X_test, eps = 1e-64) - (np.ravel(X_test**2))
                c = (Y_train - Y_train_predict) / (theta_train)
                c_clean = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                res = predict_numba(theta, X_train, Y_train , del_X_test, eps=1e-64) + theta_test * predict_numba(theta, X_train, c_clean, del_X_test, eps=1e-64)
                # res = predict_numba(theta, X_train, 2 * Y_train - Y_train_predict, del_X_test, eps=1e-64)
                RMSE = L2(res, Y_test) 
                RMSE_base = L2(res, Y_test_base)
                f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                mseLst.append(RMSE)
            theta_optimum = theta_lst[np.argmin(mseLst)]
            print(f"Selected theta is {theta_optimum}, with test rmse {min(mseLst)}, index is {np.argmin(mseLst)}")
    else:
        theta_optimum = theta_o
        print(f"Given theta is {theta_optimum}")

    # Main prediction
    del_X_train =  lagrange_multiplier_1D(theta_optimum, X_train, X_train, max_iter)
    Y_train_predict = predict_numba(theta_optimum, X_train, Y_train, del_X_train, eps=1e-64)
        
    
    theta_train = predict_sec_moment_numba(theta_optimum, X_train, del_X_train, eps=1e-64) - (np.ravel(X_train**2))
    c = (Y_train - Y_train_predict) / (theta_train)
    c_clean = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    del_x_validation =   lagrange_multiplier_1D(theta_optimum, x_validation, X_train, max_iter)
    theta_validation = predict_sec_moment_numba(theta_optimum, X_train, del_x_validation, eps=1e-64) - (np.ravel(x_validation**2))
    Y_validation_predicted = predict_numba(theta_optimum, X_train, Y_train, del_x_validation, eps=1e-64, verbose = 0) +  theta_validation * predict_numba(theta_optimum, X_train, c_clean, del_x_validation, eps=1e-64)
   
    # X2_train = np.zeros_like(X_train)
    # for idx in range (len(X_train)):
    #     X2_train[idx] = predict_quadratic_numba(0.5/theta_optimum, X_train , del_X_train[idx], eps=1e-64)
    # theta_temp_train = X2_train - X_train**2 
    
    del_x_validation_1 = lagrange_multiplier_1D(theta_optimum, x_validation, X_train, 0)
    del_x_validation_2 = lagrange_multiplier_1D(theta_optimum, x_validation, X_train, max_iter)

    n_val = len(x_validation)
    # First derivative estimation (using a symmetric linear prediction)
    x_validation_temp_1 = np.zeros_like(x_validation)
    x_validation_temp_2 = np.zeros_like(x_validation)
    x_predicted_1 = np.zeros_like(x_validation)
    x_predicted_2 = np.zeros_like(x_validation)
    for idx in range(n_val):
        a1 = 1
        x_predicted_1[idx] = predict_linear_numba(0.5/theta_optimum, X_train, del_x_validation_1[idx], eps=1e-64)
        x_predicted_2[idx] = predict_linear_numba(0.5/theta_optimum, X_train, del_x_validation_2[idx], eps=1e-64)
        delta_x = x_predicted_1[idx] - x_predicted_2[idx]
        # print(f"{idx} -> {delta_x}, delx1 = {del_x_validation_1[idx]}, delx2 = {del_x_validation_2[idx]}")
        counter = 0
        eps_temp = 0
        # Adjust if the difference is numerically zero.
        while np.abs(delta_x) < 1e-15:
            # print(f"delta_x too small (< 1e-15) for point {idx}, recalculating...")
            eps_temp = effective_dx * (counter + 1) * 1e-2
            x_predicted_1[idx] = predict_linear_numba(0.5/theta_optimum, X_train, x_validation[idx] + eps_temp, eps=1e-64)
            x_predicted_2[idx] = predict_linear_numba(0.5/theta_optimum, X_train, x_validation[idx] - eps_temp, eps=1e-64)
            delta_x = x_predicted_1[idx] - x_predicted_2[idx]
            counter += 1
            a1 = 0
        x_validation_temp_1[idx] = (1 - a1) * (x_validation[idx] + eps_temp) + a1 * del_x_validation_1[idx] 
        x_validation_temp_2[idx] = (1 - a1) * (x_validation[idx] - eps_temp) + a1 * del_x_validation_2[idx] 
    print(f"-------------------------------------")
    
    theta_validation_1 = predict_sec_moment_numba(theta_optimum, X_train, x_validation_temp_1, eps=1e-64) - (np.ravel(x_validation**2))
    theta_validation_2 = predict_sec_moment_numba(theta_optimum, X_train, x_validation_temp_2, eps=1e-64) - (np.ravel(x_validation**2))

    Y_validation_predicted_1 = predict_numba(theta_optimum, X_train,  Y_train,
                                             x_validation_temp_1, eps=1e-64) +  0*theta_validation_1 * predict_numba(theta_optimum, X_train, c_clean, del_x_validation_1, eps=1e-64)
    Y_validation_predicted_2 = predict_numba(theta_optimum, X_train,  Y_train,
                                             x_validation_temp_2, eps=1e-64) +  0*theta_validation_2 * predict_numba(theta_optimum, X_train, c_clean, del_x_validation_1, eps=1e-64)


    # Second derivative estimation


    x_predicted = np.zeros_like(x_validation)
    x2_predicted = np.zeros_like(x_validation)
    x2_predicted_1 = np.zeros_like(x_validation)
    x2_predicted_2 = np.zeros_like(x_validation)
    theta_temp = np.zeros_like(Y_validation_predicted_1)
    theta_predict = np.zeros_like(x_validation)
    Y_validation_predicted_theta1 = predict_numba(theta_optimum, X_train, Y_train, del_x_validation, eps=1e-64)
    for idx in range(n_val):
        x_predicted[idx] = predict_linear_numba(0.5/theta_optimum, X_train, del_x_validation[idx], eps=1e-64)
        x2_predicted[idx] = predict_quadratic_numba(0.5/theta_optimum, X_train, del_x_validation[idx], eps=1e-64)
        theta_temp[idx] = -x_validation[idx]**2 + x2_predicted[idx]
    

    # theta_predict = predict_numba(theta_optimum, X_train, theta_temp_train.T, del_x_validation, eps = 1e-64)

    x2_predicted_1 = - (x_validation**2).flatten() + predict_numba(theta_optimum, X_train, (X_train.T)**2, x_validation_temp_1, eps=1e-64)
    x2_predicted_2 = - (x_validation**2).flatten() + predict_numba(theta_optimum, X_train, (X_train.T)**2, x_validation_temp_2, eps=1e-64)


    d2FdX2_predicted = 2 *  (Y_validation_predicted_theta1 - Y_validation) / (theta_temp.flatten()-theta_predict.flatten())
    
    # print(f"------scale = {scale}-----------")
    # for idx in range(len(x_validation)):
    #     print(f"for x = {x_validation[idx]}, x2 = {x_validation[idx]**2}, y = {Y_validation_base[idx]} = {x_validation[idx]**2}")
    #     print('Iter 0')
    #     print(f"phi0 - x^2_1 = {Y_validation_predicted_1[idx]- predict_numba(theta_optimum, X_train, (X_train.T)**2, x_validation_temp_1, eps=1e-64)[idx]} ")
    #     print('Iter 1')
    #     print(f"phi1 - x^2_2 = {Y_validation_predicted_2[idx] - predict_numba(theta_optimum, X_train, (X_train.T)**2, x_validation_temp_2, eps=1e-64)[idx]} ")
    #     print(f"But c = {0.5 *d2FdX2_predicted[idx]}")
    #     print(f"----------------------------------")





    # First derivative dFdX estimation from symmetric predictions.
    dFdX_predicted = np.zeros_like(x_validation)


    for idx in range(n_val):
        # delta_x = np.round(x_predicted_1[idx] - x_predicted_2[idx],15)
        # term_1  = np.round(Y_validation_predicted_1[idx] - Y_validation_predicted_2[idx],15)
        # term_2 =  np.round(x2_predicted_1[idx]-x2_predicted_2[idx],15)
        
        delta_x = x_predicted_1[idx] - x_predicted_2[idx]
        term_1  = Y_validation_predicted_1[idx] - Y_validation_predicted_2[idx]
        term_2 =  x2_predicted_1[idx]-x2_predicted_2[idx]

        num = term_1 - 0.5 *d2FdX2_predicted[idx]*(term_2)
        
        b = num / delta_x
        cx = d2FdX2_predicted[idx]*x_validation[idx]
        dFdX_predicted[idx] =  b + cx
        # print(f"{idx}, {num},{term_2}, {delta_x}")
        # print(f"for {idx}, x = {x_validation[idx]}, num = {num}, term2 = {term_2}, den = {delta_x}, b = {b}, 2cx = {cx} ")
        # print(f"for {idx}, x = {x_validation[idx]}, term1 = {term_1}, term2 = {term_2} ")

    print("Complete!")
    return Y_validation_predicted/s, dFdX_predicted/s, d2FdX2_predicted/s, theta_optimum, del_x_validation



def derivatives_1D_matrix(validation_data, noise_token, training_data=None, theta_o=None, scale = None):
    print(f"\nrunning implicit")
    max_iter = 30
    dim = 1
    test_size_fraction = 0.1
    np.random.seed(25)
    if scale is not None:
        s = scale
    else:
        s = (np.loadtxt('./temp/temp_scale.txt'))[0]
    # Split data: if no separate training_data is provided, use validation_data.
    if  training_data is not None:
        data = training_data
        data[:, 1:] *= s
        validation_data[:,1:] *= s
    else:
        data = validation_data 
        data[:, 1:] *= s


    while True:
        train_data, test_data = train_test_split(data, test_size=test_size_fraction, shuffle=True)
        X_test = test_data[:, :dim]
        
        if np.any(X_test == np.max(data[:,0])) or np.any(X_test == np.min(data[:,0])):
            continue  # Retry
        else:
            break  # Success



    N = train_data.shape[0] + test_data.shape[0]
    X_train = train_data[:, :dim]
    X_test = test_data[:, :dim]
    x_validation = validation_data[:, :dim]
    Y_train_base = train_data[:, dim]
    Y_test_base = test_data[:, dim]
    Y_validation_base = validation_data[:, dim]
    if noise_token == 1:
        Y_train = train_data[:, dim+1]
        Y_test = test_data[:, dim+1]
        Y_validation = validation_data[:, dim+1]
    else:
        Y_train = Y_train_base
        Y_test = Y_test_base
        Y_validation = Y_validation_base

    # Obtain candidate theta values.
    theta_lst, effective_dx = rbf_get_theta(X_train, maX_iter=10, threshold_input=None, init_mode='knn')
    # print(f"theta: {theta_lst}")

    # Theta optimization.
    mseLst = []
    if theta_o is None:
        with open('./results/thetalist_1D.txt', 'w') as f:
            f.write('# theta   test_rmse   base_rmse\n')
            for theta in theta_lst:
                del_X_train = lagrange_multiplier_1D(theta, X_train, X_train, max_iter)
                del_X_test = lagrange_multiplier_1D(theta, X_test, X_train, max_iter)
                Y_train_predict = predict_numba(theta, X_train, Y_train, del_X_train, eps=1e-64)
                theta_train = predict_sec_moment_numba(theta, X_train, del_X_train, eps = 1e-64) - (np.ravel(X_train**2))
                theta_test = predict_sec_moment_numba(theta, X_train, del_X_test, eps = 1e-64) - (np.ravel(X_test**2))
                c = (Y_train - Y_train_predict) / (theta_train)
                c_clean = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                res = predict_numba(theta, X_train, Y_train , del_X_test, eps=1e-64) + theta_test * predict_numba(theta, X_train, c_clean, del_X_test, eps=1e-64)
                # res = predict_numba(theta, X_train, 2 * Y_train - Y_train_predict, del_X_test, eps=1e-64)
                RMSE = L2(res, Y_test) 
                RMSE_base = L2(res, Y_test_base)
                f.write(f'{theta:.3e} {RMSE:.3e} {RMSE_base:.3e}\n')
                mseLst.append(RMSE)
            theta_optimum = theta_lst[np.argmin(mseLst)]
            print(f"Selected theta is {theta_optimum}, with test rmse {min(mseLst)}, index is {np.argmin(mseLst)}")
    else:
        theta_optimum = theta_o
        print(f"Given theta is {theta_optimum}")

    # Main prediction
    del_X_train =  lagrange_multiplier_1D(theta_optimum, X_train, X_train, max_iter)
    Y_train_predict = predict_numba(theta_optimum, X_train, Y_train, del_X_train, eps=1e-64)
        
    
    theta_train = predict_sec_moment_numba(theta_optimum, X_train, del_X_train, eps=1e-64) - (np.ravel(X_train**2))
    c = (Y_train - Y_train_predict) / (theta_train)
    c_clean = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    del_x_validation =   lagrange_multiplier_1D(theta_optimum, x_validation, X_train, max_iter)
    theta_validation = predict_sec_moment_numba(theta_optimum, X_train, del_x_validation, eps=1e-64) - (np.ravel(x_validation**2))
    Y_validation_predicted = predict_numba(theta_optimum, X_train, Y_train, del_x_validation, eps=1e-64, verbose = 0) +  theta_validation * predict_numba(theta_optimum, X_train, c_clean, del_x_validation, eps=1e-64)

    
    x_input_1 =  lagrange_multiplier_1D(theta_optimum, x_validation, X_train, max_iter)

    # x_input_2 =  lagrange_multiplier_1D(theta_optimum, x_validation, X_train, 1)
    x_input_2 =  x_input_1 + 0.025*np.sqrt(theta_optimum)

    # x_input_3 =  lagrange_multiplier_1D(theta_optimum, x_validation, X_train, max_iter)
    x_input_3 =  x_input_1 - 0.025*np.sqrt(theta_optimum)



    # Precompute sizes
    n_val = len(x_validation)
    # Initialize all A_ij and B_ij from predict_numba
    A_12 = predict_numba(theta_optimum, X_train, X_train.T,          x_input_1)
    A_13 = predict_numba(theta_optimum, X_train, (X_train**2).T,       x_input_1)

    A_22 = predict_numba(theta_optimum, X_train, X_train.T,          x_input_2)
    A_23 = predict_numba(theta_optimum, X_train, (X_train**2).T,       x_input_2)

    A_32 = predict_numba(theta_optimum, X_train, X_train.T,          x_input_3)
    A_33 = predict_numba(theta_optimum, X_train, (X_train**2).T,       x_input_3)

    B_11 = predict_numba(theta_optimum, X_train, Y_train,          x_input_1)
    B_21 = predict_numba(theta_optimum, X_train, Y_train,          x_input_2)
    B_31 = predict_numba(theta_optimum, X_train, Y_train,          x_input_3)
        # Output arrays
    d2FdX2_predicted = np.zeros_like(Y_validation_predicted)
    dFdX_predicted = np.zeros_like(Y_validation_predicted)


    eps = 1e-15  # tiny diagonal shift

    # Loop over validation points
    for n in range(n_val):
        # Build A matrix
        A_matrix = np.array([
            [1.0, A_12[n], A_13[n]],
            [1.0, A_22[n], A_23[n]],
            [1.0, A_32[n], A_33[n]],
        ], dtype=float)

        # Build B vector
        B_vector = np.array([
            B_11[n],
            B_21[n],
            B_31[n]
        ], dtype=float)
        # print(f"-----------{n}--------------")
        # print(f"{A_matrix}")
        # print(f"{B_vector}")
        # print(np.linalg.det(A_matrix))

        counter = 2
        while np.linalg.det(A_matrix) == 0 :
            print(f"singular matrix for point {n}, recalculating with counter {counter}...")
            x_input_2_new =  x_input_1[n] + 0.01*counter*np.sqrt(theta_optimum)
            x_input_3_new =  x_input_1[n] - 0.01*counter*np.sqrt(theta_optimum)
            
            A_22_new = predict_linear_numba(theta_optimum, X_train, X_train.T,          x_input_2_new)
            A_23_new = predict_quadratic_numba(theta_optimum, X_train, (X_train**2).T,       x_input_2_new)
            A_32_new = predict_linear_numba(theta_optimum, X_train, X_train.T,          x_input_3_new)
            A_33_new = predict_quadratic_numba(theta_optimum, X_train, (X_train**2).T,       x_input_3_new)


            A_matrix = np.array([
            [1.0, A_12[n], A_13[n]],
            [1.0, A_22_new[0], A_23_new[0]],
            [1.0, A_32_new[0], A_33_new[0]],], dtype=float)
            counter+=1






        try:
            const_vector = np.linalg.solve(A_matrix, B_vector)
        except np.linalg.LinAlgError:
            # Regularize matrix if singular
            A_matrix = A_matrix + eps * np.eye(A_matrix.shape[0])
            print(np.linalg.det(A_matrix))

            const_vector = np.linalg.solve(A_matrix, B_vector)

        # Store predictions
        d2FdX2_predicted[n] = 2*const_vector[2]
        dFdX_predicted[n]  = const_vector[1] + 2 * const_vector[2] * x_validation[n]

        # print(f"a: {const_vector[0]}, b: {const_vector[1]}, c: {const_vector[2]}")



    print("Complete!")
    return Y_validation_predicted/s, dFdX_predicted/s, d2FdX2_predicted/s, theta_optimum, del_X_train, del_x_validation




