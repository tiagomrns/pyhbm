# %%

from dynamical_system import *
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

from pyhbm import *
#%%

"""system = ElementStEP(P=1.0, c=1000e-10)


HarmonicBalanceMethod(
    first_order_ode = system ,
    harmonics = [1,3,5,7,9],
).solve_and_continue(
    maximum_number_of_solutions = 10000,
    angular_frequency_range = [0.99995*system.omega0, 1.00005*system.omega0], 
    solver_kwargs = {
        "maximum_iterations": 20, 
        "absolute_tolerance": system.P * 1e-8
    },
    step_length_adaptation_kwargs = {
        "base": 2, 
        "maximum_step_length": 5e-5, 
        "minimum_step_length": 5e-12,  
        "goal_number_of_iterations": 3
    }
).plot_FRF(degrees_of_freedom=[0,1], harmonic=1, reference_omega=system.omega0)

plt.show()"""

system = ElementStEP_MORFE(P=1e0, c=0e-5, nonlinear=False, preconditioner=1e0)

HarmonicBalanceMethod(
    first_order_ode = system,
    harmonics = [1,3,5,-1,-3,-5],
    step_length_adaptation = BiExponentialAdaptation,
).solve_and_continue(
    maximum_number_of_solutions = 50000,
    angular_frequency_range = [0.9995*system.omega0, 1.0005*system.omega0], 
    solver_kwargs = {
        "maximum_iterations": 10, 
        "absolute_tolerance": 1e-9
    },
    step_length_adaptation_kwargs = {
        "base_increase": 4, 
        "base_decrease": 2,
        "maximum_step_length": 5e-4, 
        "minimum_step_length": 5e-20,
        "goal_number_of_iterations": 3
    }
).plot_FRF(degrees_of_freedom=0, harmonic=1)#, reference_omega=system.omega0)

#plt.show()


# %%
