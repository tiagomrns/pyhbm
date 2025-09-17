# %%

from dynamical_system import *
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../src')))

from pyhbm import *
#%%

system = ElementStEPConservative()

solver = HarmonicBalanceMethod(
    first_order_ode = system, 
    harmonics = [1,3,5,7,9], 
    predictor = TangentPredictorTwo, 
)

initial_omega = system.omega0
first_harmonic = np.array([[-1.0],[1.0j]])
initial_amplitude = 1e-6
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * initial_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega=0.0)

solution_set = solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 10000, 
    angular_frequency_range = [0.5*initial_omega, 1.5*initial_omega], 
    solver_kwargs = {
        "maximum_iterations": 200,
        "absolute_tolerance": 1e-12
    }, 
    step_length_adaptation_kwargs = {
        "base": 2.0, 
        "initial_step_length": 1e-8, 
        "maximum_step_length": 5e-6, 
        "minimum_step_length": 5e-12, 
        "goal_number_of_iterations": 3
    },
    predictor_kwargs = {
        #"rcond": None #1e-10 # In the tangent predictor, singular values s smaller than rcond * max(s) are considered zero.
    }
)

import numpy as np
from matplotlib import pyplot as plt

#for point in solution_set.fourier:
#    plt.plot(Fourier.adimensional_time_samples, point.time_series[:,0])
#plt.show()

solution_set.plot_FRF(degrees_of_freedom=0, reference_omega = system.omega0)

results_path = r"C:\Users\jpvar\Documents\TESE\code\DPIM-21-August\Joana_Pereira\Joana_Pereira\pyhbm\examples\StEP\single_element\MORFE\conservative\output\results.out"
solution_set.save(results_path)