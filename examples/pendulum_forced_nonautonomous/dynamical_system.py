#%%
import numpy as np
from numpy import cos, sin, array, concatenate
from pyhbm.dynamical_system import FirstOrderODE

#%%
class PendulumForced(FirstOrderODE):
	"""
	Class that implements the dynamics
	
	udot = omega u' = v
	vdot = omega v' = -k*sin(u) -c*v + P*cos(tau)
	
	where:
	- u is the displacement
	- v is the velocity
	- k is the gravity/length coefficient [T^-2]
	- c is the damping coefficient per unit mass [T^-1]
	- P is the amplitude of the external force per unit mass [L T^-2]
	- tau is the adimensional time, defined as tau = omega * t
	- omega is the frequency of the external force
	- t is the physical time
	- f(z, tau) is the force vector, where z = [u, v] is the state vector
	- zdot = omega z' = f(z, tau)
	
	"""
	def __init__(self, c=0.1, k=1.0, P=1.0):
		"""
		Initializes the Duffing oscillator parameters.

		:param c: Damping coefficient per unit mass [T^-1]
		:param k: Stiffness coefficient per unit mass [T^-2]
		:param P: Amplitude of the external force per unit mass [L T^-2]
		"""
		self.c = c
		self.k = k
		self.omega_resonance_linear = np.sqrt(k)
		self.P = P # also serves as reference force level
		self.linear_coefficient = array([[0.0, 1.0], [0.0, -c]])
		self.dimension = self.linear_coefficient.shape[0] # 1 dimensional in second order and 2 dimensional in first order
		self.polynomial_degree = 11

	def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Calculates the external forcing term.

		:param adimensional_time: Time at which to evaluate the external force
		:return: External force array
		"""
		zeros = np.zeros_like(adimensional_time)
		force_ext = self.P * cos(adimensional_time)
		return array([[zeros, force_ext]]).transpose()

	def linear_term(self, state: np.ndarray) -> np.ndarray:
		"""
		Calculates the linear term.

		:param state: State vector
		:return: Linear term array
		"""
		return self.linear_coefficient @ state

	def nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Calculates the nonlinear term.

		:param state: State vector
		:return: Nonlinear term array
		"""
		u = state[..., 0:1, :]  # Select first element along the second-to-last axis
		zeros = np.zeros_like(u)
		fnl = -self.k * sin(u) #* array(cos(adimensional_time))[...,np.newaxis,np.newaxis]
		return concatenate((zeros, fnl), axis=-2)

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
		"""
		Computes the Jacobian of the nonlinear term.

		:param state: State vector
		:return: Jacobian of the nonlinear term
		"""
		u = state[..., 0:1, :]  # Select first element along the second-to-last axis
		zeros = np.zeros_like(u)
		dfnldx = -self.k * cos(u) #* array(cos(adimensional_time))[...,np.newaxis,np.newaxis]  # Correct coefficient for cubic nonlinearity
		jacobian1 = np.concatenate((zeros, zeros), axis=-1)
		jacobian2 = np.concatenate((dfnldx, zeros), axis=-1)
		return concatenate((jacobian1, jacobian2), axis=-2)
	
	def jacobian_parameters(self, 
							state: np.ndarray, 
							adimensional_time: np.ndarray,
							output_c=False, 
							output_k=False, 
							output_beta=False, 
							output_P=False) -> np.ndarray:
		"""
		Computes the Jacobian w.r.t the parameters c, k, beta and P.

		:param state: State vector
		:param adimensional_time: Time for evaluating external force
		:param output_c: Boolean to decide whether to compute and output the jacobian w.r.t c
		:param output_k: Boolean to decide whether to compute and output the jacobian w.r.t k
		:param output_beta: Boolean to decide whether to compute and output the jacobian w.r.t beta
		:param output_P: Boolean to decide whether to compute and output the jacobian w.r.t P
		:return: Jacobian w.r.t the parameters
		"""
		jacobian_c, jacobian_k, jacobian_beta, jacobian_P = None, None, None, None
		
		u = state[..., 0:1, :]  # Select first element along the second-to-last axis
		zeros = np.zeros_like(u)
		
		if output_c:
			# Select second element along the second-to-last axis
			v = state[..., 1:2, :]
			# concatenate along rows to form a column
			jacobian_c = concatenate((zeros, -v), axis=-2) 
		
		if output_k:
			# concatenate along rows to form a column
			jacobian_k = concatenate((zeros, -u), axis=-2) 
		
		if output_beta:
			# concatenate along rows to form a column
			jacobian_beta = concatenate((zeros, -np.power(u, 3)), axis=-2)
		
		if output_P:
			jacobian_P = array([[np.zeros_like(adimensional_time), cos(adimensional_time)]]).transpose()
		
		return jacobian_c, jacobian_k, jacobian_beta, jacobian_P