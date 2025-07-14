# %%

from dynamical_system import *
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *
#%%

archbeam = BeamStEP(P=1, c=1e-4)  # Create an instance of ArchBeam

harmonics =  [1,3]  # Define harmonics

archbeam_solver = HarmonicBalanceMethod(
    first_order_ode = archbeam, 
    harmonics = harmonics, 
    corrector_solver = NewtonRaphson, 
    corrector_parameterization = OrthogonalParameterization, 
    predictor = TangentPredictorOne, 
    step_length_adaptation = ExponentialAdaptation
)

relative_tolerance = 1e-9

solver_kwargs = {
    "maximum_iterations": 20, 
    "absolute_tolerance": archbeam.P * relative_tolerance 
}

step_length_adaptation_kwargs = {
    "base": 2, 
    "user_step_length": 1e-5, 
    "max_step_length": 1e-3, 
    "min_step_length": 5e-10, 
    "goal_number_of_iterations": 3
}

amplitude = 0.0
omega = 1.0002*archbeam.k
z1 = amplitude * np.array([[1],[1j*omega]]) * 0.5 * Fourier.number_of_time_samples
zz = np.zeros_like(z1)
zz_fill = [zz for _ in harmonics[1:]]
initial_guess = FourierOmegaPoint(Fourier(array([z1] + zz_fill)), omega)

initial_reference_direction = FourierOmegaPoint(Fourier(array([zz] + zz_fill)), omega=-1)

import time
t0 = time.time()

solution_set = archbeam_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 10000, 
    omega_range = [0.9999*archbeam.k, 1.0007*archbeam.k], 
    solver_kwargs = solver_kwargs, 
    step_length_adaptation_kwargs = step_length_adaptation_kwargs
)

t1 = time.time()
print("total solving time =", t1-t0, "seconds")

from matplotlib import pyplot as plt
#plt.ion()

solution_set.plot_FRF(degree_of_freedom=1, harmonic=1, reference_omega=archbeam.k)
print(len(solution_set.iterations))

results_path = "examples/arch_beam_ssm/results.out"
solution_set.save(results_path)
