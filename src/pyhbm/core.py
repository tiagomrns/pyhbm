#%%
import numpy as np
from numpy import array, vstack, hstack
from numpy.linalg import norm
from matplotlib import pyplot as plt
import pickle
from time import time as current_time

from .numerical_continuation.corrector_step import *
from .numerical_continuation.predictor_step import *
from .frequency_domain import *

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

	def plot_FRF(self, degrees_of_freedom:list, harmonic:int = None, reference_omega:float =None, yscale='linear', xscale='linear', show=True, **kwargs):
     
		for dof in array([degrees_of_freedom]).ravel():
			if harmonic is None:
				# compute l2 norm of the Fourier coefficients
				harmonic_amplitude = norm(array([fourier.coefficients for fourier in self.fourier])[:, :, dof, 0], axis=1)*2/Fourier.number_of_time_samples
				plt.ylabel(r"$||\mathbf{Q}||_{DoF=%d}$" % (dof))
			else:
				index = list(Fourier.harmonics).index(harmonic) # index in the list of harmonics of the value of the harmonic
				harmonic_amplitude = abs(array([fourier.coefficients for fourier in self.fourier])[:, index, dof, 0])*2/Fourier.number_of_time_samples
				plt.ylabel(r"$|Q_{%d, %d}|$" % (harmonic, dof))

		if reference_omega is None:
			plt.plot(self.omega, harmonic_amplitude, **kwargs)
			plt.xlabel(r"$\omega$")
		else:
			omega = array(self.omega)/reference_omega
			plt.plot(omega, harmonic_amplitude, **kwargs)
			plt.xlabel(r"$\omega/\omega_0$")
		
		if yscale == 'log':
			plt.yscale('log')

		if xscale == 'log':
			plt.xscale('log')

		if show:
			plt.show()

	def save(self, path: str, 
          harmonic_amplitude = True, 
          amplitude = True, 
          angular_frequency = True,
          fourier_coefficients = False, 
          time_series = False, 
          adimensional_time_samples = False,
          iterations = False, 
          step_length = False,
          MATLAB_compatible = False):
  
		solution_set = {}

		if harmonic_amplitude:
			solution_set["harmonic_amplitude"] = \
   				abs(array([fourier.coefficients[..., 0] for fourier in self.fourier]))*2/Fourier.number_of_time_samples
       
		if amplitude:
			solution_set["amplitude"] = array([np.max(abs(fourier.time_series), axis=0) for fourier in self.fourier])
   
		if angular_frequency:
			solution_set["angular_frequency"] = self.omega.copy()

		if fourier_coefficients:
			solution_set["fourier_coefficients"] = array([fourier.coefficients.copy() for fourier in self.fourier])
   
		if time_series:
			solution_set["time_series"] = array([fourier.time_series.copy() for fourier in self.fourier])
   
		if adimensional_time_samples:
			solution_set["adimensional_time_samples"] = Fourier.adimensional_time_samples
   
		if iterations:
			solution_set["iterations"] = self.iterations.copy()
   
		if step_length:
			solution_set["step_length"] = self.step_length.copy()
  
		if MATLAB_compatible:
			from scipy.io import savemat
			savemat(path, solution_set)
		else:
			with open(path, 'wb') as handle:
				pickle.dump(solution_set, handle)

class HarmonicBalanceMethod:
	def __init__(self, first_order_ode: FirstOrderODE, 
				harmonics: np.ndarray, 
				corrector_solver = NewtonRaphson, 
				corrector_parameterization: CorrectorParameterization = OrthogonalParameterization, 
				predictor: Predictor = TangentPredictorOne, 
				step_length_adaptation: StepLengthAdaptation = ExponentialAdaptation):
     
		# Update dependencies related to harmonics and polynomial degree
		HarmonicBalanceMethod.update_dependencies(harmonics, first_order_ode.polynomial_degree)
		
		# Initialize the frequency domain ODE from the time domain ODE
		self.freq_domain_ode = FrequencyDomainFirstOrderODE_Real(first_order_ode) if first_order_ode.is_real_valued \
      		else FrequencyDomainFirstOrderODE_Complex(first_order_ode)

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
		solution, iterations, success, jacobian = self.solver(
			func = self.freq_domain_ode.compute_residue_RI, 
			jacobian = self.freq_domain_ode.compute_jacobian_of_residue_RI, 
			**solver_kwargs
		).solve(initial_guess, return_jacobian=True)
		
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
		parameterization = self.parameterization.compute_parameterization(x.to_RI_omega())
		return vstack((residue, parameterization))

	def extended_jacobian(self, x: FourierOmegaPoint):
		"""
		Compute the extended Jacobian, including parameterization derivatives.
		"""
		jacobian = self.freq_domain_ode.compute_jacobian_of_residue_RI(x)
		derivative_omega = self.freq_domain_ode.compute_derivative_wrt_omega_RI(x.fourier)
		parameterization = self.parameterization.compute_jacobian_parameterization(x.to_RI_omega())
		return vstack((hstack((jacobian, derivative_omega)), parameterization))

	def solve_and_continue(
		self, 
		maximum_number_of_solutions, 
		angular_frequency_range, 
		solver_kwargs: dict, 
		step_length_adaptation_kwargs: dict,
		predictor_kwargs: dict = {},
  		initial_guess: FourierOmegaPoint = None, 
		initial_reference_direction: FourierOmegaPoint = None, 
		#save_predicted_solutions: bool = False,
	) -> SolutionSet:

		t0 = current_time()
  
		#predicted_solutions = []

		# Sort the omega range for continuation
		angular_frequency_range.sort()

		# Set up the solver for extended system (residue + parameterization)
		solver_kwargs["absolute_tolerance"] *= np.sqrt(2) / Fourier.number_of_time_samples
  
		solver = self.solver(
			func = self.extended_residue, 
			jacobian = self.extended_jacobian, 
			**solver_kwargs
		)

		# Initialize step length adaptation
		step_length_adaptation = self.step_length_adaptation(**step_length_adaptation_kwargs)
  
		# Initialize the initial guess if not provided
		if initial_guess is None:
			initial_guess = self.zero_initialization(omega=angular_frequency_range[0])
		
  		# Initialize predictor vector
		if initial_reference_direction is not None:
			reference_direction = FourierOmegaPoint.to_RI_omega_static(initial_reference_direction)
		else: 
			reference_direction = self.zero_initialization(omega=1.0).to_RI_omega()

		# Solve the first system for a fixed frequency
		# FourierOmegaPoint, int, bool, np.ndarray
		solution, iterations, success, jacobian = self.solve_fixed_frequency(initial_guess, **solver_kwargs)
		solution_set = SolutionSet(solution, iterations, step_length_adaptation.step_length)

		if not success:
			print("\nTerminate: solver failure at initial solution (empty solution set)")
			return solution_set

		for solution_number in range(1, maximum_number_of_solutions):
			
			previous_solution: FourierOmegaPoint = solution
      
			# Predict the next solution using the predictor
   
			if self.predictor.autonomous: # remove the phase_shift_direction: np.ndarray
				phase_shift_direction = previous_solution.adimensional_time_derivative_RI()
				predictor_kwargs["remove_direction"] = phase_shift_direction / norm(phase_shift_direction)
   
			predictor_vector: np.ndarray = self.predictor.compute_predictor_vector(
       			step_length = step_length_adaptation.step_length,
				jacobian = jacobian[:self.freq_domain_ode.real_dimension],
				reference_direction = reference_direction,
				**predictor_kwargs
         	)
   
			if predictor_vector is None:
				print(f"\nTerminate: predictor failure after {solution_number} solutions")
				print(f"Current omega: {predicted_solution.omega}")
				print("Total solving time:", current_time()-t0, "seconds")
				return solution_set
			
			predicted_solution: FourierOmegaPoint = previous_solution + predictor_vector
			#if save_predicted_solutions:
			#	predicted_solutions.append(predicted_solution)
   
			# Set up corrector parameterization
			self.parameterization = self.corrector_parameterization(
				predictor_vector = predictor_vector, 
				predicted_solution = FourierOmegaPoint.to_RI_omega_static(predicted_solution)
			)

			# Solve the extended system (corrector step)
   			# FourierOmegaPoint, int, bool, np.ndarray
			solution, iterations, success, jacobian = solver.solve(predicted_solution, return_jacobian=True)
   
			if not success:
				print(f"\nTerminate: solver failure after {solution_number} solutions")
				print(f"Current omega: {predicted_solution.omega}, step length: {step_length_adaptation.step_length}")
				print("Total solving time:", current_time()-t0, "seconds")
				return solution_set#, predicted_solutions

			solution_set.append(solution, iterations, step_length_adaptation.step_length)

			progress = max(\
       			(solution.omega-angular_frequency_range[0])/(angular_frequency_range[-1]-angular_frequency_range[0]), \
				solution_number/maximum_number_of_solutions)

			print("progress {:.3f} %".format(100*progress), f"\titerations {iterations}", "\tΔω {:.2e}".format(predictor_vector[-1,0]), end="\r")

			# Update step length based on the corrector iterations
			step_length_adaptation.update_step_length(iterations)

			# Check if the omega is within the specified range
			if  not (angular_frequency_range[0] <= solution.omega <= angular_frequency_range[-1]):
				print(f"\nTerminate: outside frequency range after {solution_number+1} solutions")
				print("Total solving time:", current_time()-t0, "seconds")
				return solution_set#, predicted_solutions

			# Update the reference predictor vector for the next continuation step
			reference_direction = FourierOmegaPoint.to_RI_omega_static(solution - previous_solution)

		print("\nTerminate: maximum number of solutions reached")
		print(f"Current omega: {predicted_solution.omega}")
		print("Total solving time:", current_time()-t0, "seconds")
		return solution_set#, predicted_solutions

	def zero_initialization(self, omega):
		return FourierOmegaPoint.zero_amplitude(dimension=self.freq_domain_ode.ode.dimension, omega=omega)