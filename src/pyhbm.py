#%%
import numpy as np
from numpy import array, vstack, hstack
from numpy.linalg import norm
from matplotlib import pyplot as plt
import pickle
from time import time as current_time

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

	def plot_FRF(self, degrees_of_freedom:list, harmonic:int = None, reference_omega:float =None, yscale='linear', xscale='linear'):
     
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

	def save(self, path: str):
		
		harmonic_amplitude = abs(array([fourier.coefficients[..., 0] for fourier in self.fourier]))*2/Fourier.number_of_time_samples
		
		solution_set = {
			"harmonic_amplitude": harmonic_amplitude,
			"fourier_coefficients": array([fourier.coefficients.copy() for fourier in self.fourier]),
   			"time_series": array([fourier.time_series.copy() for fourier in self.fourier]),
      		"adimensional_time_samples": Fourier.adimensional_time_samples,
			"omega": self.omega.copy(),
			"iterations": self.iterations.copy(),
			"step_length": self.step_length.copy()
		}
  
		with open(path, 'wb') as handle:
			pickle.dump(solution_set, handle)

class HarmonicBalanceMethod:
	def __init__(self, first_order_ode, 
				harmonics: np.ndarray, 
				corrector_solver = NewtonRaphson, 
				corrector_parameterization: CorrectorParameterization = OrthogonalParameterization, 
				predictor: Predictor = TangentPredictorOne, 
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
	) -> SolutionSet:

		t0 = current_time()

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
			print("\nTerminate: solver failure at initial solution")
			return solution_set

		
  
		# print("progress {:.3f} %".format(100*(solution.omega-omega_range[0])/(omega_range[-1]-omega_range[0])), "    iterations", iterations, end="\r")

		for _ in range(maximum_number_of_solutions):
			
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
			
			predicted_solution: FourierOmegaPoint = previous_solution + predictor_vector
   
			# Set up corrector parameterization
			self.parameterization = self.corrector_parameterization(
				predictor_vector = predictor_vector, 
				predicted_solution = FourierOmegaPoint.to_RI_omega_static(predicted_solution)
			)

			# Solve the extended system (corrector step)
   			# FourierOmegaPoint, int, bool, np.ndarray
			solution, iterations, success, jacobian = solver.solve(predicted_solution, return_jacobian=True)
			solution_set.append(solution, iterations, step_length_adaptation.step_length)

			if not success:
				print("\nTerminate: solver failure")
				return solution_set

			print("progress {:.3f} %"\
         			.format(100*(solution.omega-angular_frequency_range[0])/(angular_frequency_range[-1]-angular_frequency_range[0])),\
        			 "\titerations", iterations, end="\r")

			# Update step length based on the corrector iterations
			step_length_adaptation.update_step_length(iterations)

			# Check if the omega is within the specified range
			if  not (angular_frequency_range[0] <= solution.omega <= angular_frequency_range[-1]):
				print("\nTerminate: outside frequency range.\nTotal number of points: ", len(solution_set.iterations), "\nTotal solving time:", current_time()-t0, "seconds")
				return solution_set

			# Update the reference predictor vector for the next continuation step
			reference_direction = FourierOmegaPoint.to_RI_omega_static(solution - previous_solution)

		print("\nTerminate: maximum number of solutions reached", "\nTotal solving time:", current_time()-t0, "seconds")
		return solution_set

	def _solve_and_continue(
		self, 
		maximum_number_of_solutions, 
		angular_frequency_range, 
		solver_kwargs: dict, 
		step_length_adaptation_kwargs: dict,
		predictor_kwargs: dict = {},
		initial_guess: FourierOmegaPoint = None, 
		initial_reference_direction: FourierOmegaPoint = None, 
	) -> SolutionSet:
		t0 = current_time()
		
		# Initialize the continuation process
		angular_frequency_range.sort()
		solver, step_length_adaptation, solution, jacobian, reference_direction, solution_set = \
			self._initialize_continuation(
				angular_frequency_range,
				solver_kwargs,
				step_length_adaptation_kwargs,
				initial_guess,
				initial_reference_direction
			)
		
		if not solution_set.success:
			print("\nTerminate: solver failure at initial solution")
			return solution_set

		# Perform continuation steps
		solution_set = self._perform_continuation_steps(
			maximum_number_of_solutions,
			angular_frequency_range,
			predictor_kwargs,
			solver,
			step_length_adaptation,
			solution,
			jacobian,
			reference_direction,
			solution_set,
			t0
		)
		
		return solution_set

	def _initialize_continuation(
		self,
		angular_frequency_range,
		solver_kwargs,
		step_length_adaptation_kwargs,
		initial_guess,
		initial_reference_direction
	):
		"""Initialize solver, adaptation, and compute initial solution."""
		angular_frequency_range.sort()
		
		solver_kwargs["absolute_tolerance"] *= np.sqrt(2) / Fourier.number_of_time_samples
		solver = self.solver(
			func=self.extended_residue,
			jacobian=self.extended_jacobian,
			**solver_kwargs
		)
		
		step_length_adaptation = self.step_length_adaptation(**step_length_adaptation_kwargs)
		
		if initial_guess is None:
			initial_guess = self.zero_initialization(omega=angular_frequency_range[0])
		
		if initial_reference_direction is not None:
			reference_direction = FourierOmegaPoint.to_RI_omega_static(initial_reference_direction)
		else:
			reference_direction = self.zero_initialization(omega=1.0).to_RI_omega()
		
		solution, iterations, success, jacobian = self.solve_fixed_frequency(initial_guess, **solver_kwargs)
		solution_set = SolutionSet(solution, iterations, step_length_adaptation.step_length)
		solution_set.success = success
		
		return solver, step_length_adaptation, solution, jacobian, reference_direction, solution_set

	def zero_initialization(self, omega):
		return FourierOmegaPoint.zero_amplitude(dimension=self.freq_domain_ode.ode.dimension, omega=omega)

	def _perform_continuation_steps(
		self,
		maximum_number_of_solutions,
		angular_frequency_range,
		predictor_kwargs,
		solver,
		step_length_adaptation,
		solution,
		jacobian,
		reference_direction,
		solution_set,
		t0
	):
		"""Perform the continuation steps until completion criteria are met."""
		for _ in range(maximum_number_of_solutions):
			previous_solution = solution
			
			# Predict next solution
			predictor_vector, predicted_solution = self._compute_prediction(
				previous_solution,
				step_length_adaptation.step_length,
				jacobian,
				reference_direction,
				predictor_kwargs
			)
			
			# Set up corrector parameterization
			self.parameterization = self.corrector_parameterization(
				predictor_vector=predictor_vector,
				predicted_solution=FourierOmegaPoint.to_RI_omega_static(predicted_solution)
			)
			
			# Solve extended system
			solution, iterations, success, jacobian = solver.solve(predicted_solution, return_jacobian=True)
			solution_set.append(solution, iterations, step_length_adaptation.step_length)
			
			if not success:
				print("\nTerminate: solver failure")
				return solution_set

			self._print_progress(solution.omega, angular_frequency_range, iterations)
			
			# Update step length
			step_length_adaptation.update_step_length(iterations)
			
			# Check frequency range
			if not (angular_frequency_range[0] <= solution.omega <= angular_frequency_range[-1]):
				print(f"\nTerminate: outside frequency range.\nTotal number of points: {len(solution_set.iterations)}"
					f"\nTotal solving time: {current_time()-t0} seconds")
				return solution_set
			
			# Update reference direction
			reference_direction = FourierOmegaPoint.to_RI_omega_static(solution - previous_solution)
		
		print(f"\nTerminate: maximum number of solutions reached\nTotal solving time: {current_time()-t0} seconds")
		return solution_set

	def _compute_prediction(
		self,
		previous_solution,
		step_length,
		jacobian,
		reference_direction,
		predictor_kwargs
	):
		"""Compute the predictor vector and predicted solution."""
		if self.predictor.autonomous:
			phase_shift_direction = previous_solution.adimensional_time_derivative_RI()
			predictor_kwargs["remove_direction"] = phase_shift_direction / norm(phase_shift_direction)
		
		predictor_vector = self.predictor.compute_predictor_vector(
			step_length=step_length,
			jacobian=jacobian[:self.freq_domain_ode.real_dimension],
			reference_direction=reference_direction,
			**predictor_kwargs
		)
		
		predicted_solution = previous_solution + predictor_vector
		return predictor_vector, predicted_solution

	def _print_progress(self, omega, angular_frequency_range, iterations):
		"""Print progress information."""
		progress = 100 * (omega - angular_frequency_range[0]) / (angular_frequency_range[-1] - angular_frequency_range[0])
		print(f"progress {progress:.3f} %\titerations {iterations}", end="\r")