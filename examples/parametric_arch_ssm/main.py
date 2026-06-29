# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from itertools import product

from dynamical_system import *

from pyhbm import *

theta_list = [0.0] #[-1.0, -0.5, 0.0, 0.5, 0.75, 1.0]
z_order_list = [7]

for (theta, z_truncation_order) in product(theta_list, z_order_list):
    
    print(f"\nAnalysis for theta={theta}, z_truncation_order={z_truncation_order}")
    arch = ParametricArchBackbone(theta=theta, z_truncation_order=z_truncation_order)
    #arch = ArchBackbone10mm()

    arch_solver = HarmonicBalanceMethod(
        first_order_ode = arch, 
        harmonics = [1,3,5,7,9,11,13], 
        predictor = TangentPredictorTwo, # Use TangentPredictorTwo for autonomous systems
    )

    # Define the initial guess after defining the harmonics of the HarmonicBalanceMethod
    initial_omega = -arch.omega0
    first_harmonic = np.array([[1],[1j]])
    initial_amplitude = 1e-6
    initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * initial_amplitude, omega=initial_omega)
    initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega=0.0)

    solution_set = arch_solver.solve_and_continue(
        initial_guess = initial_guess, 
        initial_reference_direction = initial_reference_direction, 
        maximum_number_of_solutions = 7000, 
        angular_frequency_range = [-0.1 * arch.omega0, -10 * arch.omega0], 
        solver_kwargs = {
            "maximum_iterations": 200,
            "absolute_tolerance": 1e-7
        }, 
        step_length_adaptation_kwargs = {
            "base": 2, 
            "initial_step_length": 1e-2, 
            "maximum_step_length": 1e0, 
            "minimum_step_length": 5e-6, 
            "goal_number_of_iterations": 3
        },
        predictor_kwargs = {
            "rcond": 1e-5, # In the tangent predictor, singular values s smaller than rcond * max(s) are considered zero.
        },
        maximum_predictor_corrector_loops_per_solution = 1,
    )

    from pyhbm import plot_FRF
    plot_FRF(solution_set, degrees_of_freedom=0, show=False)
    
    from pyhbm import save_solution_set
    file_name = f"parametric_arch_ssm_theta_{theta}_z_truncation_order_{z_truncation_order}.h5"
    path = "examples/parametric_arch_ssm/results/" + file_name
    save_solution_set(solution_set, path)
    print(f"Saved solution set to {path}")

from matplotlib import pyplot as plt
plt.show()

