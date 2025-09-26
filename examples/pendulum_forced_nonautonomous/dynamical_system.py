#%%
import numpy as np
from numpy import cos, sin, array, concatenate

class DynamicalSystem:
	"""
	Base class for dynamical systems.
	Class that implements the dynamics
	zdot = omega z' = f(z, tau) 
	tau = omega * t
	
	"""
 
	is_real_valued = True
 
	def __init__(self):
		self.linear_coefficient: int = 1
		self.dimension: int = 1
		self.polynomial_degree: int = 1
		
	def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
		raise NotImplementedError("This method should be overridden by subclasses.")

	def linear_term(self, state: np.ndarray) -> np.ndarray:
		raise NotImplementedError("This method should be overridden by subclasses.")

	def nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		raise NotImplementedError("This method should be overridden by subclasses.")

	def all_terms(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		raise NotImplementedError("This method should be overridden by subclasses.")

#%%
class DuffingForced(DynamicalSystem):
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
		self.polynomial_degree = 3

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
		fnl = -self.k * np.sin(u) #* array(cos(adimensional_time))[...,np.newaxis,np.newaxis]
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
		dfnldx = -self.k * np.cos(u) #* array(cos(adimensional_time))[...,np.newaxis,np.newaxis]  # Correct coefficient for cubic nonlinearity
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


if __name__ == '__main__':
	#%% Test the Duffing class
	
	duffing = DuffingForced(c=0.1, k=1, beta=0.5, P=2)

	state = array([[2.0], [0.0]])  # some state [x, v]
	tau = np.pi  # adimensional time or the phase

	# Print outputs for various terms
	#print("linear coefficient ", duffing.linear_coefficient.shape, " = \n", duffing.linear_coefficient)
	#print("linear term ", duffing.linear_term(state).shape, " = \n", duffing.linear_term(state))
	print("nonlinear term ", duffing.nonlinear_term(state, tau).shape, " = \n", duffing.nonlinear_term(state, tau))
	#print("jacobian nonlinear_term ", duffing.jacobian_nonlinear_term(state, tau).shape, " = \n", duffing.jacobian_nonlinear_term(state, tau))
	#print("external term ", duffing.external_term(tau).shape, " = \n", duffing.external_term(tau))
	#print("all terms = \n", duffing.all_terms(state, tau))
	jacobians_wrt_parameters = concatenate(duffing.jacobian_parameters(state, tau, True, True, True, True), axis=-1)
	print("sequence of jacobians w.r.t parameters", jacobians_wrt_parameters .shape, "\n", 
			np.round(jacobians_wrt_parameters ,9))

	#%% Test with vectorized state
	number_of_time_samples = 16
	state_vectorized = np.hstack((np.arange(number_of_time_samples).reshape(number_of_time_samples, 1, 1), 
									1+np.zeros((number_of_time_samples,)).reshape(number_of_time_samples, 1, 1))) # some sequence of states

	tau_vectorized = np.linspace(0, 2*np.pi, number_of_time_samples, endpoint=False)

	print("sequence of states", state_vectorized.shape, "\n", state_vectorized)
	print("sequence of nonlinear terms", duffing.nonlinear_term(state_vectorized, tau_vectorized).shape, "\n", 
			np.round(duffing.nonlinear_term(state_vectorized, tau_vectorized),9))
	#print("sequence of external terms", duffing.external_term(tau_vectorized).shape)
	print("sequence of jacobians of the nonlinear terms", duffing.jacobian_nonlinear_term(state_vectorized, tau_vectorized).shape, "\n", 
			np.round(duffing.jacobian_nonlinear_term(state_vectorized, tau_vectorized),9))

	sequence_of_jacobians_wrt_parameters = concatenate(duffing.jacobian_parameters(state_vectorized, tau_vectorized, True, True, True, True), axis=-1)
	print("sequence of jacobians w.r.t parameters", sequence_of_jacobians_wrt_parameters .shape, "\n", 
			np.round(sequence_of_jacobians_wrt_parameters ,9))
