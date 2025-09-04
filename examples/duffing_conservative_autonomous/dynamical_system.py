#%%
import numpy as np
from numpy import cos, sin, array, concatenate

#%%
class DuffingConservative(object):
	"""
	Class that implements the dynamics

	udot = omega u' = v
	vdot = omega v' = -k*u - beta*(u**3)

	"""
	def __init__(self, k=1.0, beta=0.1):
		"""
		Initializes the Duffing oscillator parameters.

		:param c: Damping coefficient per unit mass [T^-1]
		:param k: Stiffness coefficient per unit mass [T^-2]
		:param beta: Nonlinearity coefficient per unit mass [L^-2 T^-2]
		:param P: Amplitude of the external force per unit mass [L T^-2]
		"""
		self.k = k
		self.beta = beta
		self.omega_resonance_linear = np.sqrt(k)
		self.linear_coefficient = array([[0.0, 1.0], [-k, 0.0]])
		self.dimension = self.linear_coefficient.shape[0] # 1 dimensional in second order and 2 dimensional in first order
		self.polynomial_degree = 3
    
	def external_term(self, adimensional_time: np.ndarray):
		zeros = np.zeros_like(adimensional_time)
		return array([[zeros, zeros]]).transpose()

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
		fnl = -self.beta * np.power(u, 3)
		return concatenate((zeros, fnl), axis=-2)

	def all_terms(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Combines all terms (linear, nonlinear, external) to compute the total force.

		:param state: State vector
		:return: Total force array
		"""
		return self.linear_term(state) + self.nonlinear_term(state, adimensional_time)

	def jacobian_nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Computes the Jacobian of the nonlinear term.

		:param state: State vector
		:return: Jacobian of the nonlinear term
		"""
		u = state[..., 0:1, :]  # Select first element along the second-to-last axis
		zeros = np.zeros_like(u)
		dfnldx = -3 * self.beta * np.power(u, 2)
		jacobian1 = np.concatenate((zeros, zeros), axis=-1) # along columns
		jacobian2 = np.concatenate((dfnldx, zeros), axis=-1) # along columns
		return concatenate((jacobian1, jacobian2), axis=-2) # along row

	def jacobian_parameters(self, 
							state: np.ndarray, 
							output_k=False, 
							output_beta=False) -> np.ndarray:
		"""
		Computes the Jacobian w.r.t the parameters k and beta.

		:param state: State vector
		:return: Jacobian w.r.t the parameters
		"""
		jacobian_k, jacobian_beta = None, None
		
		u = state[..., 0:1, :]  # Select first element along the second-to-last axis
		zeros = np.zeros_like(u)
		
		if output_k:
			# concatenate along rows to form a column
			jacobian_k = concatenate((zeros, -u), axis=-2) 
		
		if output_beta:
			# concatenate along rows to form a column
			jacobian_beta = concatenate((zeros, -np.power(u, 3)), axis=-2)
		
		return jacobian_k, jacobian_beta

    
if __name__ == '__main__':
	#%% Test the Duffing class

	duffing = DuffingConservative(k=0.5, beta=0.1)

	state = array([[2.0], [0.0]])  # some state [u, v]
	tau = np.pi  # adimensional time or the phase

	# Print outputs for various terms
	print("linear coefficient ", duffing.linear_coefficient.shape, " = \n", duffing.linear_coefficient)
	print("linear term ", duffing.linear_term(state).shape, " = \n", duffing.linear_term(state))
	print("nonlinear term ", duffing.nonlinear_term(state, tau).shape, " = \n", duffing.nonlinear_term(state, tau))
	print("jacobian nonlinear_term ", duffing.jacobian_nonlinear_term(state, tau).shape, " = \n", duffing.jacobian_nonlinear_term(state, tau))
	print("all terms ", duffing.all_terms(state, tau).shape, " = \n", duffing.all_terms(state, tau))
	print("jacobian w.r.t. parameters = \n", duffing.jacobian_parameters(state, True))

	#%% Test with vectorized state
	number_of_time_samples = 16
	state_vectorized = np.hstack((np.arange(number_of_time_samples).reshape(number_of_time_samples, 1, 1), 
								np.zeros((number_of_time_samples,)).reshape(number_of_time_samples, 1, 1))) # some sequence of states
 
	tau_vectorized = np.linspace(0, 2*np.pi, number_of_time_samples, endpoint=False)

	print("sequence of states", state_vectorized.shape, "\n", state_vectorized)
	print("sequence of nonlinear terms", duffing.nonlinear_term(state_vectorized, tau_vectorized).shape, "\n", 
		np.round(duffing.nonlinear_term(state_vectorized, tau_vectorized),9))
	print("sequence of jacobians of the nonlinear terms", duffing.jacobian_nonlinear_term(state_vectorized, tau_vectorized).shape, "\n", 
		np.round(duffing.jacobian_nonlinear_term(state_vectorized, tau_vectorized),9))
	print("sequence of jacobians w.r.t parameters = \n", 
		duffing.jacobian_parameters(state_vectorized))