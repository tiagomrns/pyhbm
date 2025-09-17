#%%
from matplotlib import pyplot as plt
import pickle
import numpy as np
from math import comb
from itertools import product

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from frequency_domain import Fourier

#%%
# path = "./examples/StEP_87_elements_COMSOL_with_my_realification/results.out"
path ="./StEP_single_element_UEL_with_my_realification_3_order/results.out"
#path = "./results.out"
with open(path, 'rb') as handle:
	solution_set = pickle.load(handle)

# %%

class BCoefficientCalculator:
    @staticmethod
    def multi_index_binomial(a, m):
        res = 1
        for ai, mi in zip(a, m):
            res *= comb(ai, mi)
        return res

    @staticmethod
    def compute_B_coefficients(A_coeffs, multi_indices):
        dimension = len(multi_indices[0]) // 2 # the rows index the state variable # the columns have the global index of the monomial
        num_terms = len(multi_indices)
        B_coeffs = {}
        
        for idx in range(num_terms):
            alpha_beta = multi_indices[idx]
            alpha = alpha_beta[:dimension]
            beta = alpha_beta[dimension:]
            A_alpha_beta = A_coeffs[idx]
            
            m_ranges = [range(ai + 1) for ai in alpha]
            for m in product(*m_ranges):
                m = np.array(m)
                n_ranges = [range(bi + 1) for bi in beta]
                for n_vec in product(*n_ranges):
                    n_vec = np.array(n_vec)
                    delta = m + n_vec
                    gamma = alpha + beta - m - n_vec
                    
                    # Convert to tuple of integers for hashability
                    key = tuple(map(int, np.concatenate([gamma, delta])))
                    
                    n_sum = sum(n_vec)
                    m_sum = sum(m)
                    coeff = (A_alpha_beta * 
                             BCoefficientCalculator.multi_index_binomial(alpha, m) * 
                             BCoefficientCalculator.multi_index_binomial(beta, n_vec) * 
                             (1j) ** (m_sum - n_sum))
                    
                    if key in B_coeffs:
                        B_coeffs[key] += coeff
                    else:
                        B_coeffs[key] = coeff
        
        sorted_keys = sorted(B_coeffs.keys(), key=lambda x: (sum(x),) + tuple(-xi for xi in x))
        B_coeffs_sorted = np.array([B_coeffs[key] for key in sorted_keys])
        multi_indices_B = [np.array(key) for key in sorted_keys]
        
        return B_coeffs_sorted, multi_indices_B



class PolynomialConverter:
    @staticmethod
    def multi_index_power(z, alpha):
        # Vectorized version for z of shape (N, dim)
        result = np.ones(z.shape[0])
        for i in range(z.shape[1]):
            result *= z[:, i] ** alpha[i]
            #print(f"result{result}")
        return result.reshape(-1,1)

    @staticmethod
    def evaluate_f_xy(x, y, coefficients, multi_indices):
        # x and y are now arrays of shape (N, dim)
        N = x.shape[0]
        dim = x.shape[1]
        f_vals = np.zeros((N, 1), dtype=complex)
        
        for mindex, coeff in zip(multi_indices, coefficients):
            gamma = mindex[:dim]
            delta = mindex[dim:]
            # Compute power products for entire batch
            x_power = PolynomialConverter.multi_index_power(x, gamma)
            y_power = PolynomialConverter.multi_index_power(y, delta)
            f_vals += coeff * x_power * y_power

        # Convert to real if imaginary part is negligible
        if np.all(np.abs(f_vals.imag) < 1e-15):
            return f_vals.real.astype(float)
        return f_vals

    @staticmethod
    def convert_B_to_nonlinear_polynomial(gamma_delta, B):
        terms = []
        for mindex, val in zip(gamma_delta, B):
            #CHANGE IF WE HAVE MORE VARIABLES
            # Only include nonlinear terms (sum > 1) that involve both x and y variables
            if (mindex[0] != 0 or mindex[2] != 0) and sum(mindex) > 1:
                term_parts = []
                for i, power in enumerate(mindex):
                    if power == 0:
                        continue
                    term_parts.append(f"x{i}" if power == 1 else f"(x{i}**{power})")
                if term_parts:
                    terms.append(f"({val}) * {' * '.join(term_parts)}")
        return "fnl = \\\n            " + " +\\\n            ".join(terms) if terms else "fnl = 0"

    @staticmethod
    def convert_B_to_force_polynomial(gamma_delta, B):
        terms = []
        for mindex, val in zip(gamma_delta, B):
            #CHANGE IF WE HAVE MORE VARIABLES
            # Only include terms with pure y variables (x0=0, x2=0) and at least linear
            if (mindex[0] == 0 and mindex[2] == 0) and sum(mindex) >= 1:
                term_parts = []
                for i, power in enumerate(mindex):
                    if power == 0:
                        continue
                    term_parts.append(f"x{i}" if power == 1 else f"(x{i}**{power})")
                if term_parts:
                    terms.append(f"({val}) * {' * '.join(term_parts)}")
        return "fext = \\\n            " + " +\\\n            ".join(terms) if terms else "fext = 0"

#%%
# Which DOF (row index) you want
dof = 0 

# Load the matrix from the Python-format file
with open("./StEP_single_element_UEL_with_my_realification_3_order/Wu_matrix.txt") as f:
    # Evaluate the file to get the array (safe only if you trust the file)
    data_str = f.read()
    Wu_matrix = eval(data_str.split('=', 1)[1].strip())


# Select the row corresponding to this DOF
W_u_coeffs = Wu_matrix[dof, :] 
print(f"W_u_coefficients:{W_u_coeffs.shape}")

############
dof_v = 0

# Load the matrix from the Python-format file
with open("./StEP_single_element_UEL_with_my_realification_3_order/Wv_matrix.txt") as f:
    # Evaluate the file to get the array (safe only if you trust the file)
    data_str = f.read()
    Wv_matrix = eval(data_str.split('=', 1)[1].strip())


# Select the row corresponding to this DOF
W_v_coeffs = Wv_matrix[dof_v, :]
print(f"W_v_matrix:{Wv_matrix.shape}")

###################
dof_z = 2
# Load the matrix from the Python-format file
with open("./StEP_single_element_UEL_with_my_realification_3_order/Wu_matrix.txt") as f:
    # Evaluate the file to get the array (safe only if you trust the file)
    data_str = f.read()
    Wu_matrix_z = eval(data_str.split('=', 1)[1].strip())

# Select the row corresponding to this DOF
W_u_coeffs_z = Wv_matrix[dof_z, :]

print(f"coefficients: {W_u_coeffs_z.shape}")

multi_indices = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [2,0,0,0],
        [1,1,0,0],
        [1,0,1,0],
        [1,0,0,1],
        [0,2,0,0],
        [0,1,1,0],
        [0,1,0,1],
        [0,0,2,0],
        [0,0,1,1],
        [0,0,0,2],
        [3,0,0,0],
        [2,1,0,0],
        [2,0,1,0],
        [2,0,0,1],
        [1,2,0,0],
        [1,1,1,0],
        [1,1,0,1],
        [1,0,2,0],
        [1,0,1,1],
        [1,0,0,2],
        [0,3,0,0],
        [0,2,1,0],
        [0,2,0,1],
        [0,1,2,0],
        [0,1,1,1],
        [0,1,0,2],
        [0,0,3,0],
        [0,0,2,1],
        [0,0,1,2],
        [0,0,0,3]
])[:, [0,2,1,3]]

###################################
# Compute B coefficients
B_W_u, gamma_delta = BCoefficientCalculator.compute_B_coefficients(
    W_u_coeffs, multi_indices
)

# Compute B coefficients
B_W_v, gamma_delta_v = BCoefficientCalculator.compute_B_coefficients(
    W_v_coeffs, multi_indices
)

# Compute B coefficients
B_W_z, gamma_delta_z = BCoefficientCalculator.compute_B_coefficients(
    W_u_coeffs_z, multi_indices
)
###################


# Filter near-zero coefficients
"""threshold = 1e-15
nonzero_mask = np.abs(B_W_u) > threshold
# `B_W_u` is a set of coefficients that are computed using the `compute_B_coefficients` method of the
# `BCoefficientCalculator` class. These coefficients are used to represent the nonlinear terms in a
# polynomial expression that describes the displacement mapping function. The `B_W_u` coefficients are
# calculated based on the input `W_u_coeffs` and `multi_indices` arrays, which represent the
# coefficients and multi-indices of the terms in the polynomial respectively. These coefficients are
# then used in the `evaluate_displacement_mapping` function to evaluate the displacement mapping for
# each solution point.
B_W_u = B_W_u[nonzero_mask]
gamma_delta = np.array(gamma_delta)[nonzero_mask]
#print(f"B coefficients: {B_W_u}")
print(f"gamma delta:{gamma_delta.shape}")

B_W_v = B_W_v[nonzero_mask]
gamma_delta_v = np.array(gamma_delta_v)[nonzero_mask]

B_W_z = B_W_z[nonzero_mask]
gamma_delta_z = np.array(gamma_delta_z)[nonzero_mask]"""
#%%

P = 1.0

# index = solution_point, time, dof
x_all = solution_set["time_series"][:,:,0,0]
print(f"x_all:{x_all.shape}")
y_all = solution_set["time_series"][:,:,1,0]
c = P * np.cos(solution_set["adimensional_time_samples"])
print(f"cos:{c.shape}")
s = P * np.sin(solution_set["adimensional_time_samples"])

def evaluate_displacement_mapping(x,c,y,s, B, gamma_delta): # vectorized function
    x_total = np.column_stack((x, c))
    y_total = np.column_stack((y, s))
    return PolynomialConverter.evaluate_f_xy(x_total, y_total, B, gamma_delta)


"""def evaluate_displacement_mapping(x, c, y, s, coefficients, multi_indices):
    # Convert all input arrays to column vectors
    
    N = x.shape[0]
    f_vals = np.zeros(N, dtype=complex)
    
    for mindex, coeff in zip(multi_indices, coefficients):
        a1, a2, a3, a4 = mindex
        f_vals += coeff * (x**a1) * (c**a2) * (y**a3) * (s**a4)

    return f_vals.real"""

list_of_fourier = []

"""def compute_polymomial_for_each_solution():
    #max_error = 0
    for solution_point, (x,y) in enumerate(zip(x_all, y_all)):
        # print(f"x:{x}")
        # displacement_time_series = evaluate_displacement_mapping(x_total,y_total)
        displacement_time_series = evaluate_displacement_mapping(x, c, y, s, B_W_u, gamma_delta)
        #error = np.linalg.norm(displacement_time_series.imag)/np.linalg.norm(displacement_time_series)
        #if error > max_error: max_error = error

        fourier_coefficient = np.fft.rfft(displacement_time_series)
        list_of_fourier.append(np.linalg.norm(fourier_coefficient))
    #print("max error =", max_error)

compute_polymomial_for_each_solution()
plt.plot(solution_set["omega"], list_of_fourier)
plt.ylabel(r"Displacement $||U_{%d}||$"%(dof))
plt.xlabel(r"$\omega$")
plt.show()"""

#%%
x_max = np.max(x_all)
y_max = np.max(y_all)

# ... [previous code] ...

#ax = plt.axes(projection='3d')
time = solution_set["adimensional_time_samples"]
total_number_of_solutions = len(solution_set["omega"])

# Precompute blue values for all points (vectorized)
blue_vals = np.max(x_all, axis=1) / x_max  # Max over time for each point

for idx, (x, y) in enumerate(zip(x_all[::100], y_all[::100])):
    # Evaluate z and flatten to 1D
    x_u = evaluate_displacement_mapping(x, c, y, s, B_W_u, gamma_delta).ravel()
    y_v = evaluate_displacement_mapping(x, c, y, s, B_W_v, gamma_delta_v).ravel()
    z_u = evaluate_displacement_mapping(x, c, y, s, B_W_z, gamma_delta_z).ravel()
    
    # Plot with color based on max displacement
    #ax.plot3D(x_u, y_v, z_u, color=(0.0, 0.0, blue_vals[idx]))
    plt.plot(x_u, y_v)

# Plot representative points (every 100th solution point)
"""for idx, (x, y) in enumerate(zip(x_all[::10000], y_all[::10000])):
    x_u = evaluate_displacement_mapping(x, c, y, s, B_W_u, gamma_delta).ravel()
    y_v = evaluate_displacement_mapping(x, c, y, s, B_W_v, gamma_delta_v).ravel()
    z_u = evaluate_displacement_mapping(x, c, y, s, B_W_z, gamma_delta_z).ravel()
    ax.plot3D(x_u, y_v, z_u, color='r')"""

plt.show()
# %%
