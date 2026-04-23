#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import ArchBeamSSM

from pyhbm import *

#%%
archbeam = ArchBeamSSM(P=1)

solution = HarmonicBalanceMethod(
    first_order_ode = archbeam,
    harmonics = [1,3,5,7,9],
).solve_and_continue(
    maximum_number_of_solutions = 1500,
    angular_frequency_range = [0.9*archbeam.omega0, 1.1*archbeam.omega0], 
    solver_kwargs = {
        "maximum_iterations": 20, 
        "absolute_tolerance": archbeam.P * 1e-6
    },
    step_length_adaptation_kwargs = {
        "base": 2, 
        "maximum_step_length": 10.0, 
        "minimum_step_length": 5e-12, 
        "goal_number_of_iterations": 3
    }
)

from pyhbm import plot_FRF
plot_FRF(solution, degrees_of_freedom=[0,1], harmonic=1, reference_omega=archbeam.omega0)
