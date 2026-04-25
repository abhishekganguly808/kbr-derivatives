import numpy as np
from headers import thermal_interpolator as ti



# dim = int(input("Enter Dim:"))
noise_token = 0





training_data = np.loadtxt('./temp/validation_data.txt')
# flux_data = np.loadtxt('./temp/flux_data.txt')
# print(flux_predicted.flatten())

_, _, _, _, _, _ = ti.derivatives_1D_matrix(training_data, noise_token)
# _, _, _, _, _, _ = ti.derivatives_1D_PDE(training_data, noise_token)



