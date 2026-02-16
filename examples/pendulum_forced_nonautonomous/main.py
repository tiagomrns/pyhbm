# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import *
from pyhbm import *
import matplotlib.pyplot as plt

pendulum = PendulumForced(c=0.07, k=1.0, P=0.1148)  # Create an instance of PendulumForced

pendulum_solver = HarmonicBalanceMethod(
    first_order_ode = pendulum, 
    harmonics = [1,3,5,7,9,11,13], 
)

# Define the initial guess after defining the harmonics of the HarmonicBalanceMethod
initial_omega = 0.0
first_harmonic = np.array([[1],[1j*initial_omega]])
static_amplitude = pendulum.P/pendulum.k
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * static_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega=1)

solution_set = pendulum_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 1000000, 
    angular_frequency_range = [0.0, 2], 
    solver_kwargs = {
        "maximum_iterations": 200, 
        "absolute_tolerance": pendulum.P * 1e-6
    }, 
    step_length_adaptation_kwargs = {
        "base": 2, 
        "initial_step_length": 0.1, 
        "maximum_step_length": 0.1, 
        "minimum_step_length": 5e-6, 
        "goal_number_of_iterations": 3
    }
)

plt.title("Frequency Response Function - Forced Pendulum")
solution_set.plot_FRF(degrees_of_freedom=0)
solution_set.save("examples/pendulum_forced_nonautonomous/results.out")

# Find maximum amplitude point
index = np.argmax([np.linalg.norm(fourier.coefficients[:,0,0], axis=0) for fourier in solution_set.fourier])
plt.title("Time Series of Angle at Peak Amplitude")
plt.xlabel = "Adimensional Time"
plt.ylabel = "Pendulum angle"
plt.plot(solution_set.fourier[index].adimensional_time_samples, solution_set.fourier[index].time_series[:,0,0])
plt.show()

plt.title("Phase Plot at Peak Amplitude")
plt.xlabel = "Pendulum angle"
plt.xlabel = "Pendulum angular velocity"
plt.plot(solution_set.fourier[index].time_series[:,0,0], solution_set.fourier[index].time_series[:,1,0])
plt.show()
# %%
