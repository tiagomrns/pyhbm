# %%

from dynamical_system import *

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *

duffing = DuffingForced(c=0.01, k=1.0, beta=1, P=1.0)  # Create an instance of Duffing

duffing_solver = HarmonicBalanceMethod(
    first_order_ode = duffing, 
    harmonics = array([1,3,5,7,9]), 
    corrector_solver = NewtonRaphson, 
    corrector_parameterization = OrthogonalParameterization, 
    predictor = TangentPredictorOne, 
    step_length_adaptation = ExponentialAdaptation
)

solver_kwargs = {
    "maximum_iterations": 200, 
    "absolute_tolerance": duffing.P * 0.0001
}

step_length_adaptation_kwargs = {
    "base": 2, 
    "user_step_length": 0.1, 
    "max_step_length": 0.2, 
    "min_step_length": 5e-6, 
    "goal_number_of_iterations": 3
}

import time

t1 = time.time()

amplitude = duffing.P/duffing.k
omega = 0.1
z1 = amplitude * np.array([[1],[1j*omega]]) * 0.5 * Fourier.number_of_time_samples
zz = np.zeros_like(z1)
initial_guess = FourierOmegaPoint(Fourier(array([z1,zz,zz,zz,zz])), omega)

initial_reference_direction = FourierOmegaPoint(Fourier(array([zz,zz,zz,zz,zz])), 1)

solution_set = duffing_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 1000000, 
    omega_range = [0.1, 10], 
    solver_kwargs = solver_kwargs, 
    step_length_adaptation_kwargs = step_length_adaptation_kwargs
)

from matplotlib import pyplot as plt
# for interactive plots in Jupyter notebook
#%matplotlib widget

solution_set.plot_FRF(degree_of_freedom=0)#, yscale='log', xscale='log')

print("time elapsed =", time.time() - t1)

# %%
