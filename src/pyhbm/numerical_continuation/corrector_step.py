from numpy import vdot
from numpy.linalg import norm, solve

class NewtonRaphson(object):
	def __init__(self, func, jacobian, maximum_iterations: int, absolute_tolerance: float,
				 jacobian_update_frequency: int = 3,
				 jacobian_reuse_delta_threshold: float = 1e-3):
		self.compute_residue = func
		self.compute_jacobian = jacobian
		self.maximum_iterations = maximum_iterations
		self.absolute_tolerance = absolute_tolerance
		self.jacobian_update_frequency = jacobian_update_frequency
		self.jacobian_reuse_delta_threshold = jacobian_reuse_delta_threshold
	
	def is_converged(self) -> bool:
		return norm(self.residue) < self.absolute_tolerance
	
	def update_jacobian(self, iteration: int):
		if self._should_update_jacobian(iteration):
			self.jacobian = self.compute_jacobian(self.x)
			self._last_jacobian_update_iter = iteration
	
	def _should_update_jacobian(self, iteration: int) -> bool:
     
		if (iteration - self._last_jacobian_update_iter) >= self.jacobian_update_frequency:
			return True
		
		return norm(self.delta) >= self.jacobian_reuse_delta_threshold * norm(self.x)
	
	def compute_increment(self):
		self.delta = solve(self.jacobian, self.residue)
	
	def update_solution(self):
		self.x = self.x - self.delta
	
	def get_converged_result(self, iteration: int, return_jacobian: bool):
		if not return_jacobian:
			return self.x, iteration, True
		if iteration == 0:
			self.jacobian = self.compute_jacobian(self.x)
		return self.x, iteration, True, self.jacobian
	
	def get_failed_result(self, return_jacobian: bool):
		return (self.x, self.maximum_iterations, False, self.jacobian) if return_jacobian \
				else (self.x, self.maximum_iterations, False)
	
	def solve(self, initial_guess, return_jacobian: bool = False):
		self.x = initial_guess
		self._last_jacobian_update_iter = -self.jacobian_update_frequency
		
		for iteration in range(self.maximum_iterations):
			self.residue = self.compute_residue(self.x)
			
			if self.is_converged():
				return self.get_converged_result(iteration, return_jacobian)
			
			self.update_jacobian(iteration)
			self.compute_increment()
			self.update_solution()
		
		print(f"Newton-Raphson: maximum iterations reached ({self.maximum_iterations})")
		return self.get_failed_result(return_jacobian)

#%%

class CorrectorParameterization(object):
	def compute_parameterization(**kwargs):
		pass

	def compute_jacobian_parameterization(**kwargs):
		pass

class OrthogonalParameterization(CorrectorParameterization):
	def __init__(self, predictor_vector, predicted_solution):
		self.predictor_vector = predictor_vector
		self.predicted_solution = predicted_solution

	def compute_parameterization(self, point, *args):
		return vdot(self.predictor_vector, point - self.predicted_solution)

	def compute_jacobian_parameterization(self, *args):
		return self.predictor_vector.T

class ArcLengthParameterization(CorrectorParameterization):
	def __init__(self, last_solution, step_size):
		self.last_solution = last_solution
		self.step_size = step_size

	def compute_parameterization(self, point, *args):
		delta = point - self.last_solution
		return vdot(delta, delta) - self.step_size**2

	def compute_jacobian_parameterization(self, point, *args):
		delta = point - self.last_solution
		return 2 * delta.T
