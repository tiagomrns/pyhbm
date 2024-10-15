# %%

from dynamical_system import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *
#%%

archbeam = ArchBeamSSM(P=1)  # Create an instance of Duffing

harmonics =  2*np.arange(3)+1 # hstack((2*np.arange(20)+1, 40)) # 2*np.arange(4)+1

archbeam_solver = HarmonicBalanceMethod(
    first_order_ode = archbeam, 
    harmonics = harmonics, 
    corrector_solver = NewtonRaphson, 
    corrector_parameterization = OrthogonalParameterization, 
    predictor = TangentPredictor, 
    step_length_adaptation = ExponentialAdaptation
)

solver_kwargs = {
    "maximum_iterations": 20, 
    "absolute_tolerance": archbeam.P * 1e-3
}

step_length_adaptation_kwargs = {
    "base": 2, 
    "user_step_length": 0.01, 
    "max_step_length": 1.0, 
    "min_step_length": 5e-12, 
    "goal_number_of_iterations": 2
}

amplitude = archbeam.P/archbeam.k
omega = 0.7*archbeam.k
z1 = amplitude * np.array([[1],[1j*omega]]) * 0.5 * Fourier.number_of_time_samples
zz = np.zeros_like(z1)
zz_fill = [zz for _ in harmonics[1:]]
initial_guess = FourierOmegaPoint(Fourier(array([z1] + zz_fill)), omega)

initial_reference_direction = FourierOmegaPoint(Fourier(array([zz] + zz_fill)), 1)

import time
t0 = time.time()

solution_set = archbeam_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 1000000, 
    omega_range = [omega, 1.3*archbeam.k], 
    solver_kwargs = solver_kwargs, 
    step_length_adaptation_kwargs = step_length_adaptation_kwargs
)

t1 = time.time()
print("total solving time =", t1-t0, "seconds")

from matplotlib import pyplot as plt
#plt.ion()

solution_set.plot_FRF(degree_of_freedom=1, harmonic=1, reference_omega=archbeam.k)
results_path = "./examples/arch_bem_ssm/timed_results/new.out"
solution_set.save(results_path)
