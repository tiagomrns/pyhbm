# %%

from dynamical_system import *

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *

duffing = DuffingConservative(k=1.0, beta=0.1)

duffing_solver = HarmonicBalanceMethod(
    first_order_ode = duffing, 
    harmonics = [1,3,5,7,9], 
    predictor = TangentPredictorTwo, # Use TangentPredictorTwo for autonomous systems
)

# Define the initial guess after defining the harmonics of the HarmonicBalanceMethod
initial_omega = duffing.omega_resonance_linear
first_harmonic = np.array([[1],[1j*initial_omega]])
initial_amplitude = 1e-6
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * initial_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega=0.0)

duffing_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 1000, 
    angular_frequency_range = [0.9, 5], 
    solver_kwargs = {
        "maximum_iterations": 200,
        "absolute_tolerance": duffing.k * 1e-7
    }, 
    step_length_adaptation_kwargs = {
        "base": 2, 
        "initial_step_length": 0.01, 
        "maximum_step_length": 2.0, 
        "minimum_step_length": 5e-6, 
        "goal_number_of_iterations": 3
    },
    predictor_kwargs = {
        "rcond": 1e-6 # In the tangent predictor, singular values s smaller than rcond * max(s) are considered zero.
    }
).plot_FRF(degrees_of_freedom=0)