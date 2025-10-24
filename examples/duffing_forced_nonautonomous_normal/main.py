# %%

from dynamical_system import *

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *

duffing = DuffingForced(c=0.1, k=1.0, beta=-0.35, P=0.1)  # Create an instance of Duffing

duffing_solver = HarmonicBalanceMethod(
    first_order_ode = duffing, 
    harmonics = [1,3,5,7,9], 
)

# Define the initial guess after defining the harmonics of the HarmonicBalanceMethod
initial_omega = 0.6
first_harmonic = np.array([[1],[1j*initial_omega]])
static_amplitude = duffing.P/duffing.k
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * static_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega=1)

solution_set_reference, _ = duffing_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 100000, 
    angular_frequency_range = [0.0, 1.3], 
    solver_kwargs = {
        "maximum_iterations": 200, 
        "absolute_tolerance": duffing.P * 1e-6
    }, 
    step_length_adaptation_kwargs = {
        "base": 2, 
        "initial_step_length": 0.01, 
        "maximum_step_length": 0.01, 
        "minimum_step_length": 5e-6, 
        "goal_number_of_iterations": 3
    },
)

solution_set_reference.plot_FRF(degrees_of_freedom=0, color='k', show=False)

for idx, (fourier, omega) in enumerate(zip(solution_set_reference.fourier, solution_set_reference.omega)):
    if idx not in [500, 2000, 5500, 7500]:
        continue
    harmonic_amplitude = norm(fourier.coefficients[:, 0, 0])*2/Fourier.number_of_time_samples
    print("idx:", idx, "omega:", omega, "amplitude:", harmonic_amplitude)
    plt.plot(omega, harmonic_amplitude, color='k', marker='o', markersize=5)
    
plt.show()

for idx in [500, 2000, 5500, 7500]:
    solution = solution_set_reference.fourier[idx]
    plt.plot(solution.adimensional_time_samples, solution.time_series[:,0,0], lw=4)
    plt.ylim((-1.2, 1.2))
    plt.show()


"""solution_set, predicted_solutions = duffing_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 3500, 
    angular_frequency_range = [0.0, 1.3], 
    solver_kwargs = {
        "maximum_iterations": 200, 
        "absolute_tolerance": duffing.P * 1e-6
    }, 
    step_length_adaptation_kwargs = {
        "base": 2, 
        "initial_step_length": 0.1, 
        "maximum_step_length": 0.5, 
        "minimum_step_length": 5e-6, 
        "goal_number_of_iterations": 3
    },
    save_predicted_solutions = True
)


solution_set_reference.plot_FRF(degrees_of_freedom=0, color='k', ls='--', show=False)
solution_set.plot_FRF(degrees_of_freedom=0, color='k', linewidth=0.0, marker='.', markersize=10, show=False)
    
for idx in range(len(predicted_solutions)-1):
    ps = predicted_solutions[idx]
    harmonic_amplitude_ps = norm(ps.fourier.coefficients[:, 0, 0])*2/Fourier.number_of_time_samples
    harmonic_amplitude_s = norm(solution_set.fourier[idx+1].coefficients[:, 0, 0])*2/Fourier.number_of_time_samples
    plt.plot([ps.omega, solution_set.omega[idx+1]], [harmonic_amplitude_ps, harmonic_amplitude_s], color='r', linewidth=1.0)
    
# lines that connect the solutions to the predicted solutions
for idx in range(len(predicted_solutions)-1):
    ps = predicted_solutions[idx]
    harmonic_amplitude_ps = norm(ps.fourier.coefficients[:, 0, 0])*2/Fourier.number_of_time_samples
    harmonic_amplitude_s = norm(solution_set.fourier[idx].coefficients[:, 0, 0])*2/Fourier.number_of_time_samples
    plt.plot([ps.omega, solution_set.omega[idx]], [harmonic_amplitude_ps, harmonic_amplitude_s], color='b', linewidth=1.0)
    
for idx, predicted_solution in enumerate(predicted_solutions):
    harmonic_amplitude = norm(predicted_solution.fourier.coefficients[:, 0, 0])*2/Fourier.number_of_time_samples
    plt.plot(predicted_solution.omega, harmonic_amplitude, color='b', marker='o', markersize=5)
    

plt.show()

"""# %%
