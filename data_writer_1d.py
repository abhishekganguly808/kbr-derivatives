from headers import functions as fn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


###########################
# Grid
dim = 1
num_points = int(input("Enter the number of model training points (N): "))

n_scale = float(input("Enter noise scale (s): "))
noise_type = 'Gaussian'



# Random Grid
np.random.seed(42)
x = np.sort(np.random.rand(num_points))  # [0,1]
xmin, xmax = np.min(x), np.max(x)
# print(f"x_min: {np.min(x)}, x_max: {np.max(x)}")

# Uniform grid
# dx_eff = 1.0/(num_points-1)
# xmin = 0 
# xmax = 1 
# # x = np.linspace(xmin, xmax, num_points)
# # Uniform parameter space in [0,1]
# u = np.linspace(0.0, 1.0, num_points)

# # Arcsin mapping -> dense in middle, sparse near edges
# alpha = 0
# x = (1 - alpha) * u + alpha*(0.5 + np.arcsin(2*u - 1) / np.pi)




F = np.zeros_like(x)
dFdX = np.zeros_like(x)
d2FdX2 = np.zeros_like(x)

# F, dFdX, d2FdX2, terms = fn.Weierstrass(x)
# F, dFdX, d2FdX2 = fn.Polynomial(x,0,0,0,1,0)
F, dFdX, d2FdX2 = fn.Camel1D(x)



# Normalize
scale = np.array([np.max(np.abs(F)), np.max(np.abs(dFdX)), np.max(np.abs(d2FdX2))])
# print(f"scale: {scale}")
# with open(f"./temp/temp_scale.txt", "w") as f:
#     f.write(f"{scale.ravel()}\n")
np.savetxt(f"./temp/temp_scale.txt", scale.ravel(), header="[0] scale F\n[1] scale dFdX\n[2] scale d2FdX2")



F = fn.Normalize(F,scale[0])

dFdX = fn.Normalize(dFdX,scale[0])
d2FdX2 = fn.Normalize(d2FdX2,scale[0])
# Add Noise
Fn = fn.AddNoise(F,noise_type,n_scale)



###########################
# Store result

data = np.column_stack([x.ravel(), F.ravel(), Fn.ravel(), dFdX.ravel(), d2FdX2.ravel()] )

np.savetxt(f"./temp/function1D_{num_points}.txt", data, header="[0] X\n[1] F\n[2] F + noise\n[3] dFdX\n[4] d2FdX2")

# Regular validation Grid
# np.random.seed(41)
# n_validation = int(input("Enter the number of deployment/test points: "))
# X_validation = np.sort( np.random.rand(n_validation))




# ## Functions ########
# F_validation,dFdX_validation,d2FdX2_validation = fn.gaussian(X_validation,0.5,400)
# F_validation, dFdX_validation, d2FdX2_validation = fn.generic_initial_condition(X_validation)
# F_validation, dFdX_validation, d2FdX2_validation = fn.riemann_function(X_validation,1,0,0.5)
# F_validation, dFdX_validation, d2FdX2_validation = fn.sine(X_validation,2)

# F_validation,dFdX_validation, d2FdX2_validation = fn.Polynomial(X_validation,0,0,1,0,0)

# #### Normalization and Add Noise
# F_validation = fn.Normalize(F_validation,scale[0])


# dFdX_validation = fn.Normalize(dFdX_validation,scale[0])
# d2FdX2_validation = fn.Normalize(d2FdX2_validation,scale[0])


# Fn_validation= fn.AddNoise(F_validation,noise_type,n_scale)


# #     ##########################
# #     # Store validation Data
# #     #before running this, create a folder 'Data' that will save all the data files
# data = np.column_stack([X_validation.ravel(), F_validation.ravel(), Fn_validation.ravel(), dFdX_validation.ravel(), d2FdX2_validation.ravel()] )
# np.savetxt(f"./temp/validation_data.txt", data, header="[0] X_validation\n[1] F_validation\n[2] F_validation + noise\n[3] dFdX_validation\n[4] d2FdX2_validation")


# data = np.column_stack([X_validation.ravel(), F_validation.ravel(), Fn_validation.ravel(), dFdX_validation.ravel(), d2FdX2_validation.ravel()] )



# Read data
data = np.loadtxt(f"./temp/function1D_{num_points}.txt")
np.savetxt('./temp/validation_data.txt', data)


# # This is done because in clusters where scikit is not installed, the raw data can be split into train and test while writing
# # Read data
# num_points = data.shape[0]
# # print(f"number of points: {num_points}")

# test_size_fraction = 0.2
# np.random.seed(25)


# while True:
#         train_data, test_data = train_test_split(data, test_size=test_size_fraction, shuffle=True)
#         X_test = test_data[:, :dim]
        
#         if np.any(X_test == np.max(data[:,0])) or np.any(X_test == np.min(data[:,0])):
#             continue  # Retry
#         else:
#             break  # Success



# # Save the training and testing data to text files
# np.savetxt('./temp/train_data.txt', train_data)
# np.savetxt('./temp/test_data.txt', test_data)
