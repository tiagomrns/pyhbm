# %%

from dynamical_system import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *

duffing = Duffing(c=0.1, k=1.0, beta=0.1, P=0.18)  # Create an instance of Duffing

duffing_solver = HarmonicBalanceMethod(
    first_order_ode = duffing, 
    harmonics = array([1,3,5]), 
    corrector_solver = NewtonRaphson, 
    corrector_parameterization = OrthogonalParameterization, 
    predictor = TangentPredictor, 
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
    "goal_number_of_iterations": 1
}

amplitude = duffing.P/duffing.k
omega = 0.5
z1 = amplitude * np.array([[1],[1j*omega]]) * 0.5 * Fourier.number_of_time_samples
zz = np.zeros_like(z1)
initial_guess = FourierOmegaPoint(Fourier(array([z1,zz,zz])), omega)

initial_reference_direction = FourierOmegaPoint(Fourier(array([zz,zz,zz])), 1)

solution_set = duffing_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 1000000, 
    omega_range = [0.5, 1.50], 
    solver_kwargs = solver_kwargs, 
    step_length_adaptation_kwargs = step_length_adaptation_kwargs
)


from matplotlib import pyplot as plt
#plt.ion()

solution_set.plot_FRF(degree_of_freedom=1, harmonic=1)

# %%
