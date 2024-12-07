#%%
import numpy as np
from numpy import cos, sin, array, concatenate, unique, block, vstack, hstack, array_split
from numpy.fft import rfft, irfft, fft
from numpy.linalg import norm, solve
from matplotlib import pyplot as plt
import pickle

from numerical_continuation.corrector_step import *
from numerical_continuation.predictor_step import *
from frequency_domain import *

# Delta_RI = np.linalg.solve(self.freq_domain_ode.jacobian_residue_RI, -self.freq_domain_ode.residue_RI)

#Delta_C = Delta_RI[:self.freq_domain_ode.complex_dimension] +\
#          1j*Delta_RI[self.freq_domain_ode.complex_dimension:]

#Delta = Fourier(array(array_split(Delta_C, Fourier.number_of_harmonics)))

#initial_guess += Delta

class SolutionSet(object):
	def __init__(self, solution: FourierOmegaPoint, iterations: int, step_length: float):
		self.fourier = [solution.fourier]
		self.omega = [solution.omega]
		self.iterations = [iterations]
		self.step_length = [step_length]
		
	def append(self, solution: FourierOmegaPoint, iterations: int, step_length: float):
		self.fourier.append(solution.fourier)
		self.omega.append(solution.omega)
		self.iterations.append(iterations)
		self.step_length.append(step_length)
  
	def __len__(self):
		return len(self.solution)

	def plot_FRF(self, degree_of_freedom, harmonic=None, reference_omega=None, yscale='linear', xscale='linear'):

		if harmonic is None:
			# compute l2 norm of the Fourier coefficients
			harmonic_amplitude = norm(array([fourier.coefficients for fourier in self.fourier])[:, :, degree_of_freedom, 0], axis=1)*2/Fourier.number_of_time_samples
			plt.ylabel(r"$||\mathbf{Q}||_{DoF=%d}$" % (degree_of_freedom))
		else:
			index = list(Fourier.harmonics).index(harmonic) # index in the list of harmonics of the value of the harmonic
			harmonic_amplitude = abs(array([fourier.coefficients for fourier in self.fourier])[:, index, degree_of_freedom, 0])*2/Fourier.number_of_time_samples
			plt.ylabel(r"$|Q_{%d, %d}|$" % (harmonic, degree_of_freedom))

		if reference_omega is None:
			plt.plot(self.omega, harmonic_amplitude)
			plt.xlabel(r"$\omega$")
		else:
			omega = array(self.omega)/reference_omega
			plt.plot(omega, harmonic_amplitude)
			plt.xlabel(r"$\omega/\omega_0$")
		
		if yscale == 'log':
			plt.yscale('log')

		if xscale == 'log':
			plt.xscale('log')

		plt.show()

	def save(self, path):
		
		harmonic_amplitude = abs(array([fourier.coefficients[..., 0] for fourier in self.fourier]))*2/Fourier.number_of_time_samples
		
		solution_set = {
			"harmonic_amplitude": harmonic_amplitude,
			"fourier": self.fourier.copy(),
			"omega": self.omega.copy(),
			"iterations": self.iterations.copy(),
			"step_length": self.step_length.copy()
		}
		with open(path, 'wb') as handle:
			pickle.dump(solution_set, handle)


class HarmonicBalanceMethod:
	def __init__(self, first_order_ode, 
				harmonics: np.ndarray, 
				corrector_solver=NewtonRaphson, 
				corrector_parameterization: CorrectorParameterization = OrthogonalParameterization, 
				predictor: Predictor = TangentPredictor, 
				step_length_adaptation: StepLengthAdaptation = ExponentialAdaptation):
     
		# Update dependencies related to harmonics and polynomial degree
		HarmonicBalanceMethod.update_dependencies(harmonics, first_order_ode.polynomial_degree)
		
		# Initialize the frequency domain ODE from the time domain ODE
		self.freq_domain_ode = FrequencyDomainFirstOrderODE(first_order_ode)

		# Set up solver components
		self.solver = corrector_solver
		self.corrector_parameterization = corrector_parameterization
		self.predictor = predictor
		self.step_length_adaptation = step_length_adaptation

		# Reference force level (used for tolerance calculation)
		self.reference_force_level = norm(self.freq_domain_ode.external_term.coefficients)

	@staticmethod
	def update_dependencies(harmonics: np.ndarray, polynomial_degree: int):
		# Update global Fourier settings based on harmonics and polynomial degree
		Fourier.update_class_variables(harmonics, polynomial_degree)
		JacobianFourier.update_class_variables()

	def solve_fixed_frequency(self, initial_guess: FourierOmegaPoint, **solver_kwargs):
		"""
		Solve the system for a fixed frequency.
		Returns the solution and the Jacobian matrix.
		"""
		solver = self.solver(
			func = self.freq_domain_ode.compute_residue_RI, 
			jacobian = self.freq_domain_ode.compute_jacobian_of_residue_RI, 
			**solver_kwargs
		)
		
		solution, iterations, success, jacobian = solver.solve(initial_guess, return_jacobian=True)
		
		derivative_omega = self.freq_domain_ode.compute_derivative_wrt_omega_RI(solution.fourier)
		
		return \
			solution, \
			iterations, \
			success, \
   			hstack((jacobian, derivative_omega))
		

	def extended_residue(self, x: FourierOmegaPoint):
		"""
		Compute the extended residue, including parameterization.
		"""
		residue = self.freq_domain_ode.compute_residue_RI(x)
		parameterization = self.parameterization.compute_parameterization(FourierOmegaPoint.to_RI_omega(x))
		return vstack((residue, parameterization))

	def extended_jacobian(self, x: FourierOmegaPoint):
		"""
		Compute the extended Jacobian, including parameterization derivatives.
		"""
		jacobian = self.freq_domain_ode.compute_jacobian_of_residue_RI(x)
		derivative_omega = self.freq_domain_ode.compute_derivative_wrt_omega_RI(x.fourier)
		parameterization = self.parameterization.compute_jacobian_parameterization(FourierOmegaPoint.to_RI_omega(x))
		return vstack((hstack((jacobian, derivative_omega)), parameterization))

	def solve_and_continue(
		self, 
		initial_guess: FourierOmegaPoint, 
		initial_reference_direction, 
		maximum_number_of_solutions, 
		omega_range, 
		solver_kwargs: dict, 
		step_length_adaptation_kwargs: dict
	):

		# Sort the omega range for continuation
		omega_range.sort()

		# Set up the solver for extended system (residue + parameterization)
		solver_kwargs["absolute_tolerance"] *= np.sqrt(2) / Fourier.number_of_time_samples
		solver = self.solver(
			func = self.extended_residue, 
			jacobian = self.extended_jacobian, 
			**solver_kwargs
		)

		# Initialize step length adaptation
		step_length_adaptation = self.step_length_adaptation(**step_length_adaptation_kwargs)

		# Solve the first system for a fixed frequency
		solution, iterations, success, jacobian = self.solve_fixed_frequency(initial_guess, **solver_kwargs)
		solution_set = SolutionSet(solution, iterations, step_length_adaptation.step_length)

		if not success:
			print("\nTerminate: solver failure at initial solution")
			return solution_set

		# Initialize predictor vector and loop through solutions
		reference_predictor_vector = FourierOmegaPoint.to_RI_omega(initial_reference_direction)
  
		print("progress {:.3f} %".format(100*(solution.omega-omega_range[0])/(omega_range[-1]-omega_range[0])), "    iterations", iterations, end="\r")

		for _ in range(maximum_number_of_solutions):
			
			previous_solution = solution
      
			# Predict the next solution using the predictor
			predictor_vector = self.predictor.compute_predictor_vector(
				step_length = step_length_adaptation.step_length, 
				jacobian = jacobian[:self.freq_domain_ode.real_dimension], 
				reference_predictor_vector = reference_predictor_vector
			)
			
			predicted_solution = solution + predictor_vector
			
			# Set up corrector parameterization
			self.parameterization = self.corrector_parameterization(
				predictor_vector = predictor_vector, 
				predicted_solution = FourierOmegaPoint.to_RI_omega(predicted_solution)
			)

			# Solve the extended system (corrector step)
			solution, iterations, success, jacobian = solver.solve(predicted_solution, return_jacobian=True)
			solution_set.append(solution, iterations, step_length_adaptation.step_length)

			if not success:
				print("\nTerminate: solver failure")
				return solution_set

			print("progress {:.3f} %".format(100*(solution.omega-omega_range[0])/(omega_range[-1]-omega_range[0])), "    iterations", iterations, end="\r")

			# Update step length based on the corrector iterations
			step_length_adaptation.update_step_length(iterations)

			# Check if the omega is within the specified range
			if  not (omega_range[0] <= solution.omega <= omega_range[-1]):
				print("\nTerminate: outside omega range")
				return solution_set

			# Update the reference predictor vector for the next continuation step
			reference_predictor_vector = FourierOmegaPoint.to_RI_omega(solution - previous_solution)

		print("\nTerminate: maximum number of points reached")
		return solution_set



# %%
