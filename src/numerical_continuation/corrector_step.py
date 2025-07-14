import numpy as np
from numpy import vdot
from numpy.linalg import norm, solve

class NewtonRaphson(object):
	def __init__(self, func, jacobian, maximum_iterations: int, absolute_tolerance: float):
		self.compute_residue = func  # Function to compute residue
		self.compute_jacobian = jacobian  # Function to compute Jacobian
		self.maximum_iterations: int = maximum_iterations
		self.absolute_tolerance: float = absolute_tolerance

	def convergence_check(self) -> bool:
		# Check for convergence based on the absolute tolerance
		return norm(self.residue) < self.absolute_tolerance

	def increment(self):
		# Solve for the increment using the Jacobian and the residue
		delta: np.ndarray = solve(self.jacobian, -self.residue)
		self.x = self.x + delta  # Update the guess
  
	def converged_solution(self, iteration: int, return_jacobian: bool):
		if not return_jacobian: 
			return self.x, iteration, True
		if iteration == 0:
			self.jacobian = self.compute_jacobian(self.x)
		return self.x, iteration, True, self.jacobian
		# output jacobian only when iteration > 1 also

	def solve(self, initial_guess, return_jacobian: bool = False):
		self.x = initial_guess # E.g. FourierOmegaPoint or np.ndarray
		for iteration in range(self.maximum_iterations):
			self.residue = self.compute_residue(self.x)
			if self.convergence_check():
				return self.converged_solution(iteration, return_jacobian)

			self.jacobian = self.compute_jacobian(self.x)
			self.increment()

		print("Maximum number of iterations reached:", self.maximum_iterations)
		return (self.x, self.maximum_iterations, False, self.jacobian) if return_jacobian \
				else (self.x, self.maximum_iterations, False)
	
		# the output is solution, iterations, success boolean, jacobian 


# from scipy.optimize import newton

class CorrectorParameterization(object):
	def compute_parameterization(*kwargs):
		pass

	def compute_jacobian_parameterization(*kwargs):
		pass

class OrthogonalParameterization(CorrectorParameterization):
	def __init__(self, predictor_vector, predicted_solution, **kwargs):
		self.predictor_vector = predictor_vector
		self.predicted_solution = predicted_solution

	def compute_parameterization(self, point, *kwargs):
		return vdot(self.predictor_vector, point - self.predicted_solution)

	def compute_jacobian_parameterization(self, *kwargs):
		return self.predictor_vector.T



