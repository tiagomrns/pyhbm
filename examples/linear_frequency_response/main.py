# %%

from dynamical_system import *

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *

#%% System number 1

A = array([
    [-0.01, 4],
    [-4 ,-0.01]
])

C = array([[1,0]]).T
S = array([[1,0]]).T

HarmonicBalanceMethod(
    first_order_ode = LinearOscillator(A, C, S), 
    harmonics = [1], 
).solve_and_continue(
    maximum_number_of_solutions = 1000, 
    angular_frequency_range = [0.0, 10], 
    solver_kwargs = {
        "maximum_iterations": 200, 
        "absolute_tolerance": 1e-6
    }, 
    step_length_adaptation_kwargs = {
        "base": 2, 
        "initial_step_length": 0.1, 
        "maximum_step_length": 2.0, 
        "minimum_step_length": 5e-6, 
        "goal_number_of_iterations": 3
    }
).plot_FRF(degrees_of_freedom=0)

#%% System number 2

A = array([
    [0, 1],
    [-16 ,-0.01]
])

C = array([[0,1]]).T
S = array([[0,0]]).T

HarmonicBalanceMethod(
    first_order_ode = LinearOscillator(A, C, S), 
    harmonics = [1], 
).solve_and_continue(
    maximum_number_of_solutions = 1000, 
    angular_frequency_range = [0.0, 10], 
    solver_kwargs = {
        "maximum_iterations": 200, 
        "absolute_tolerance": 1e-6
    }, 
    step_length_adaptation_kwargs = {
        "base": 2, 
        "initial_step_length": 0.1, 
        "maximum_step_length": 2.0, 
        "minimum_step_length": 5e-6, 
        "goal_number_of_iterations": 3
    }
).plot_FRF(degrees_of_freedom=0)