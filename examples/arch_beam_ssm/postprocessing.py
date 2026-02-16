#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from matplotlib import pyplot as plt
import pickle
import numpy as np

import sys
import os
from pyhbm import Fourier

#%%
path = "./results.out"
with open(path, 'rb') as handle:
	solution_set = pickle.load(handle)

# %%

P = 1

# index = solution_point, time, dof
x_all = solution_set["time_series"][:,:,0,0]
y_all = solution_set["time_series"][:,:,1,0]
c = P * np.cos(solution_set["adimensional_time_samples"])
s = P * np.sin(solution_set["adimensional_time_samples"])

dof = 0
def evaluate_displacement_mapping(x,c,y,s, dof): # vectorized function
    # mapping_coefficients = W[0]
    # return evaluate_f_xy(mapping_coefficients, multi_indices, x,c,y,s)
    return x**2 + x**5 + 3*x*y**2 + y**3 + y**2*c + y*x + s

list_of_fourier = []

def compute_polymomial_for_each_solution():
    for solution_point, (x,y) in enumerate(zip(x_all, y_all)):
        displacement_time_series = evaluate_displacement_mapping(x,c,y,s,dof)
        fourier_coefficient = np.fft.rfft(displacement_time_series)
        list_of_fourier.append(fourier_coefficient)

compute_polymomial_for_each_solution()
plt.plot(solution_set["omega"], np.abs(np.array(list_of_fourier)[:,1]))
plt.ylabel(r"Displacement $||U_{%d}||$"%(dof))
plt.xlabel(r"$\omega$")
plt.show()

#%%
x_max = np.max(x_all)
y_max = np.max(y_all)

ax = plt.axes(projection='3d')
time = solution_set["adimensional_time_samples"]
total_number_of_solutions = len(solution_set["omega"])
for count, (x, y) in enumerate(zip(x_all, y_all)):
    blue = max(x)/x_max
    #red = max(y)/y_max
    ax.plot3D(x, y, evaluate_displacement_mapping(x,c,y,s,dof), color=(0.0, 0.0, blue))
    
for x, y in zip(x_all[::100], y_all[::100]):
    ax.plot3D(x, y, evaluate_displacement_mapping(x,c,y,s,dof), color='r', lw=0.7)

plt.show()
# %%
