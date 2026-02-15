# %%

from dynamical_system import *

from pyhbm import *

duffing = DuffingForced(c=0.009, k=1.0, beta=1.0, P=1.0)  # Create an instance of Duffing

duffing_solver = HarmonicBalanceMethod(
    first_order_ode = duffing, 
    harmonics = [1,3,5,7,9], 
)

# Define the initial guess after defining the harmonics of the HarmonicBalanceMethod
initial_omega = 0.0
first_harmonic = np.array([[1],[1j*initial_omega]])
static_amplitude = duffing.P/duffing.k
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * static_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega=1)

solution_set = duffing_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 3500, 
    angular_frequency_range = [0.0, 10], 
    solver_kwargs = {
        "maximum_iterations": 200, 
        "absolute_tolerance": duffing.P * 1e-6
    }, 
    step_length_adaptation_kwargs = {
        "base": 2, 
        "initial_step_length": 0.1, 
        "maximum_step_length": 2.0, 
        "minimum_step_length": 5e-6, 
        "goal_number_of_iterations": 3
    }
)

solution_set.plot_FRF(degrees_of_freedom=0, xscale='log', yscale='log')
# %%
