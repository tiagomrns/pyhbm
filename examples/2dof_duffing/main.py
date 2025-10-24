# %%

from dynamical_system import *

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *

clist = [0.1, 0.05, 0.02, 0.0152, 0.005]

for c in clist:

    system = System2DoF(c=c, k=1.0, beta1=1.0, beta2=0.8, r=20.0)  # Create an instance of Duffing

    solver = HarmonicBalanceMethod(
        first_order_ode = system, 
        harmonics = [1,3,5,7,9], 
    )

    # Define the initial guess after defining the harmonics of the HarmonicBalanceMethod
    initial_omega = 4.0
    first_harmonic = np.array([[0], [0], [1],[1j*initial_omega]])
    static_amplitude = 0.01 #system.P/system.k
    initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * static_amplitude, omega=initial_omega)
    initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic*0, omega=-1)

    solution_set = solver.solve_and_continue(
        initial_guess = initial_guess, 
        initial_reference_direction = initial_reference_direction, 
        maximum_number_of_solutions = 100000, 
        angular_frequency_range = [0.0, 5], 
        solver_kwargs = {
            "maximum_iterations": 200, 
            "absolute_tolerance": system.P * 1e-6
        }, 
        step_length_adaptation_kwargs = {
            "base": 2, 
            "initial_step_length": 1e-3, 
            "maximum_step_length": 0.5, 
            "minimum_step_length": 5e-6, 
            "goal_number_of_iterations": 3
        }
    )
    
    if c == clist[-2]:
        point = 500
        isola_init_omega = solution_set.omega[point]
        isola_init_fourier = solution_set.fourier[point]

    solution_set.plot_FRF(degrees_of_freedom=0, show=False)
    solution_set.save("examples/2dof_duffing/data_frequency_response_curves/frf_2dof_duffing_c=%.4f.mat" % c, MATLAB_compatible=True)
    
    if c == clist[-1]:
        initial_reference_direction = FourierOmegaPoint(fourier=isola_init_fourier, omega=1)
        
        solution_set = solver.solve_and_continue(
            initial_guess = FourierOmegaPoint(fourier=isola_init_fourier, omega=isola_init_omega), 
            maximum_number_of_solutions = 4000, 
            angular_frequency_range = [1.7, 4.1], 
            solver_kwargs = {
                "maximum_iterations": 200, 
                "absolute_tolerance": system.P * 1e-6
            }, 
            step_length_adaptation_kwargs = {
                "base": 2, 
                "initial_step_length": 1e-3, 
                "maximum_step_length": 0.5, 
                "minimum_step_length": 5e-6, 
                "goal_number_of_iterations": 3
            }
        )
        solution_set.plot_FRF(degrees_of_freedom=0, show=False)
        solution_set.save("examples/2dof_duffing/data_frequency_response_curves/frf_2dof_duffing_c=%.4f_isola.mat" % c, MATLAB_compatible=True)


plt.show()
