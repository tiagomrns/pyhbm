# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import *

from pyhbm import *

duffing = DuffingForced(c=0.009, k=1.0, beta=1.0, P=1.0)  # Create an instance of Duffing

duffing_solver = HarmonicBalanceMethod(
    first_order_ode = duffing, 
    harmonics = [1,3,5,7,9], 
)

# Define the initial guess after defining the harmonics of the HarmonicBalanceMethod
initial_omega = 0.0
first_harmonic = np.array([[1],[1j*initial_omega]]) * 1
static_amplitude = duffing.P/duffing.k
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * static_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega=1)

solution_set = duffing_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 3500, 
    angular_frequency_range = [0.0, 15.0], 
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

from pyhbm import plot_FRF

plot_FRF(solution_set, degrees_of_freedom=0, xscale='log', yscale='log', stability=True, time_domain_ode=duffing)#, show=False)

# %% Time-Domain Validation
from pyhbm import TimeDomainValidator

validator = TimeDomainValidator(duffing, integrator='RK45')

index_to_validate = 1
fourier = solution_set.fourier[index_to_validate]
omega = solution_set.omega[index_to_validate]

time_series = fourier.time_series

print(f"\nValidating solution at omega = {omega:.4f}")

result = validator.validate(
    time_series=time_series,
    omega=omega,
    multiplier_sampling_rate=40
)

print(f" Relative RMS error:\t{result.relative_rms_error:.6e}")
print(f" Relative Max error:\t{result.relative_max_error:.6e}")
print(f" Phase error:\t{result.phase_error:.6e}")

validator.plot_comparison(result, degrees_of_freedom=0, show=False)
# %%
