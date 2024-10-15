from dynamical_system import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import time

from pyhbm import *

archbeam = ArchBeamSSM(P=1)  # Create an instance of Duffing
amplitude = archbeam.P/archbeam.k
omega = 0.7*archbeam.k

# Lists of parameters to loop over
harmonics_list = [array([1, 3]), array([1, 3, 5]), array([1, 3, 5, 7]), array([1, 3, 5, 7, 9]), array([1, 3, 5, 7, 9, 11, 13]), array([1, 3, 5, 7, 9, 11, 13, 15, 17])]
tolerances_list = [1e-3, 1e-6, 1e-9]  # Change as needed
max_step_length_list = [0.1, 0.5, 1.0]  # Change as needed
goal_iterations_list = [2, 3, 4]  # Change as needed

# Base path to save results
results_dir = "./examples/arch_bem_ssm/timed_results"
summary_file_path = results_dir + "/summary_table.txt"

# Open the summary file to write the header
with open(summary_file_path, "w") as summary_file:
    summary_file.write("H\ttol\tmax_s\titer_goal\tTotal Time (s)\n")

# Loop over all parameter combinations
for harmonics in harmonics_list:
    for tolerance in tolerances_list:
        for max_step_length in max_step_length_list:
            for goal_iterations in goal_iterations_list:
                # Create unique folder for each combination of parameters
                folder_name = f"h={len(harmonics)}_tol={tolerance}_step={max_step_length}_iter={goal_iterations}"
                folder_path = os.path.join(results_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Update solver_kwargs and step_length_adaptation_kwargs with current parameters
                solver_kwargs = {
                    "maximum_iterations": 20,
                    "absolute_tolerance": archbeam.P * tolerance
                }

                step_length_adaptation_kwargs = {
                    "base": 2,
                    "user_step_length": 0.01,
                    "max_step_length": max_step_length,
                    "min_step_length": 5e-12,
                    "goal_number_of_iterations": goal_iterations
                }

                # Create solver instance with current harmonics
                archbeam_solver = HarmonicBalanceMethod(
                    first_order_ode=archbeam,
                    harmonics=harmonics,
                    corrector_solver=NewtonRaphson,
                    corrector_parameterization=OrthogonalParameterization,
                    predictor=TangentPredictor,
                    step_length_adaptation=ExponentialAdaptation
                )

                z1 = amplitude * np.array([[1],[1j*omega]]) * 0.5 * Fourier.number_of_time_samples
                zz = np.zeros_like(z1)
                zz_fill = [zz for _ in harmonics[1:]]
                initial_guess = FourierOmegaPoint(Fourier(array([z1] + zz_fill)), omega)
                initial_reference_direction = FourierOmegaPoint(Fourier(array([zz]+ zz_fill)), 1)

                # Start timing
                t0 = time.time()

                # Solve the system
                solution_set = archbeam_solver.solve_and_continue(
                    initial_guess=initial_guess,
                    initial_reference_direction=initial_reference_direction,
                    maximum_number_of_solutions=1000000,
                    omega_range=[omega, 1.3 * archbeam.k],
                    solver_kwargs=solver_kwargs,
                    step_length_adaptation_kwargs=step_length_adaptation_kwargs
                )

                # End timing
                t1 = time.time()

                # Save solving time
                total_time = t1 - t0
                with open(os.path.join(folder_path, "parameters_and_time.txt"), "w") as f:
                    f.write(f"{len(harmonics)}\t{tolerance}\t{max_step_length}\t{goal_iterations}\t{total_time:.4f}\n")

                # Save solution set in the current folder (you can adjust how to save based on your solution structure)
                # solution_set.save(os.path.join(folder_path, "solution_set.out"))  # Assuming solution_set has a save method

                # Write current run's data into the summary file
                with open(summary_file_path, "a") as summary_file:
                    summary_file.write(f"{harmonics}\t{tolerance}\t{max_step_length}\t{goal_iterations}\t{total_time:.4f}\n")


                print(f"Run saved in: {folder_path}, Solving time: {total_time:.4f} seconds")
