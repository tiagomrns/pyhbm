# %%

from dynamical_system import *

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyhbm import *

duffing = DuffingConservative(k=1.0, beta=0.1)  # Create an instance of Duffing

harmonics = [1,3,5,7,9]  # Define harmonics

duffing_solver = HarmonicBalanceMethod(
    first_order_ode = duffing, 
    harmonics = harmonics, 
    corrector_solver = NewtonRaphson, 
    corrector_parameterization = OrthogonalParameterization, 
    predictor = TangentPredictorTwo,
    step_length_adaptation = ExponentialAdaptation
)

solver_kwargs = {
    "maximum_iterations": 200, 
    "absolute_tolerance": duffing.k * 1e-4
}

step_length_adaptation_kwargs = {
    "base": 2, 
    "user_step_length": 0.01, 
    "max_step_length": 0.1, 
    "min_step_length": 5e-6, 
    "goal_number_of_iterations": 3
}

predictor_kwargs = {
    "rcond": 1e-6
}

print("duffing_solver.freq_domain_ode.harmonics =", Fourier.harmonics)

amplitude = 1e-6
omega = duffing.omega_resonance_linear
z1 = np.array([[1],[1j*omega]]) * 0.5 * Fourier.number_of_time_samples
zz = np.zeros_like(z1)
zz_fill = [zz for _ in harmonics[1:]]
initial_guess = FourierOmegaPoint(Fourier(array([z1 * amplitude] + zz_fill)), omega)
initial_reference_direction = FourierOmegaPoint(Fourier(array([z1] + zz_fill)), 0)

solution_set = duffing_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 1000, 
    omega_range = [0.01, 5], 
    solver_kwargs = solver_kwargs, 
    step_length_adaptation_kwargs = step_length_adaptation_kwargs,
    predictor_kwargs = predictor_kwargs,
)

#%%
from matplotlib import pyplot as plt
# for interactive plots in Jupyter notebook
#%matplotlib widget

solution_set.plot_FRF(degree_of_freedom=0)#, yscale='log', xscale='log')
#%%

omega_plot = 1.15
tol = 1e-3
idx = np.argwhere(abs(np.array(solution_set.omega) - omega_plot) < tol)[0,0]
omega_ref = solution_set.omega[idx] # <- For Abaqus
print(omega_ref)

fourier_ref = solution_set.fourier[idx]

Fourier.number_of_time_samples = 1000
Fourier.adimensional_time_samples = linspace(0, 2*pi, Fourier.number_of_time_samples, endpoint=False)

fourier_ref.compute_time_series()

time_series = fourier_ref.time_series[:,:,0]

#displacement = [Wu(z,r1,r2) for z,r in zip(
    # time_series, 
    # np.exp(1j*Fourier.adimensional_time_samples),
    # np.sin(-1j*Fourier.adimensional_time_samples))
#]

plt.plot(Fourier.adimensional_time_samples/omega_ref, time_series)
plt.show()

# %%
