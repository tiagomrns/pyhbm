#%%
import numpy as np
from numpy import cos, sin, array, concatenate, zeros
from pyhbm.dynamical_system import FirstOrderODE

#%%
class LinearOscillator(FirstOrderODE):
	"""
	Class that implements the dynamics
	
	xdot = omega x' = A @ x + C * cos(tau) + S * sin(tau)
	
	where:
	- x is the state column vector
	- v is the velocity
	- A is a constant matrix
	- C and S are constant column vectors
 	- omega is the frequency of the external force
	- t is the physical time. Derivative w.r.t time is xdot
	- tau is the adimensional time, defined as tau = omega * t. Derivative w.r.t tau is x'
	
	"""
 
	def __init__(self, A=array, C=array, S=array, is_real_valued = True):
		"""
		Initializes the Duffing oscillator parameters.

		:param A: constant matrix
		:param C: constant column vector
		:param S: constant column vector
		"""
		self.linear_coefficient = array(A)
		self.C, self.S = array(C)[None,...], array(S)[None,...]
		self.dimension = self.linear_coefficient.shape[0]
		self.polynomial_degree = 1
		self.is_real_valued = is_real_valued

	def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Calculates the external term.

		:param adimensional_time: Time at which to evaluate the external force
		:return: External term array
		"""
		return self.C * cos(adimensional_time)[...,None,None] + self.S * sin(adimensional_time)[...,None,None]

	def linear_term(self, state: np.ndarray) -> np.ndarray:
		"""
		Calculates the linear term.

		:param state: State vector
		:return: Linear term array
		"""
		return self.linear_coefficient @ state

	def nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		return np.zeros_like(state)

	def all_terms(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Combines all terms (linear, nonlinear, external) to compute the total force.

		:param state: State vector
		:param adimensional_time: Time for evaluating external force
		:return: Total force array
		"""
		return self.linear_term(state) + \
				self.nonlinear_term(state, adimensional_time) + \
				self.external_term(adimensional_time)

	def jacobian_nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		number_of_states = state.shape[-2]
		return zeros((len(adimensional_time), number_of_states, number_of_states))