#%%

from dynamical_system import ArchBeamSSM

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *

#%%
archbeam = ArchBeamSSM(P=1)

HarmonicBalanceMethod(
    first_order_ode = archbeam,
    harmonics = [1,3,5,7,9],
).solve_and_continue(
    maximum_number_of_solutions = 1500,
    angular_frequency_range = [0.7*archbeam.omega0, 1.3*archbeam.omega0], 
    solver_kwargs = {
        "maximum_iterations": 20, 
        "absolute_tolerance": archbeam.P * 1e-6
    },
    step_length_adaptation_kwargs = {
        "base": 2, 
        "maximum_step_length": 2.0, 
        "minimum_step_length": 5e-12, 
        "goal_number_of_iterations": 3
    }
).plot_FRF(degrees_of_freedom=[0,1], harmonic=1, reference_omega=archbeam.omega0)
