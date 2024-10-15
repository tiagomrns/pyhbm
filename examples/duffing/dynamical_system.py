#%%
import numpy as np
from numpy import cos, sin, array, concatenate


#%%
class Duffing:
  def __init__(self, c=0.1, k=1.0, beta=0.0, P=0.2):
    """
    Initializes the Duffing oscillator parameters.

    :param c: Damping coefficient per unit mass [T^-1]
    :param k: Stiffness coefficient per unit mass [T^-2]
    :param beta: Nonlinearity coefficient per unit mass [L^-2 T^-2]
    :param P: Amplitude of the external force per unit mass [L T^-2]
    """
    self.c = c
    self.k = k
    self.beta = beta
    self.P = P # also serves as reference force level
    self.linear_coefficient = array([[0.0, 1.0], [-k, -c]])
    self.dimension = self.linear_coefficient.shape[0] # 1 dimensional in second order and 2 dimensional in first order
    self.polynomial_degree = 3

    self.reference_force = P
    self.reference_length = P/k if k != 0 else 1
    self.reference_energy = self.reference_force * self.reference_length

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
    x = state[..., 0:1, :]  # Select first element along the second-to-last axis
    zeros = np.zeros_like(x)
    fnl = -self.beta * np.power(x, 3) * array(cos(adimensional_time))[...,np.newaxis,np.newaxis]
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
    x = state[..., 0:1, :]  # Select first element along the second-to-last axis
    zeros = np.zeros_like(x)
    dfnldx = -3 * self.beta * np.power(x, 2) * array(cos(adimensional_time))[...,np.newaxis,np.newaxis]  # Correct coefficient for cubic nonlinearity
    jacobian1 = np.concatenate((zeros, zeros), axis=-1)
    jacobian2 = np.concatenate((dfnldx, zeros), axis=-1)
    return concatenate((jacobian1, jacobian2), axis=-2)

  def jacobian_parameters() -> None:
    # future proofing stuff
    pass


#%% Test the Duffing class

"""duffing = Duffing(c=0, k=0, beta=10, P=2)

state = array([[2.0], [0.0]])  # some state [x, v]
tau = np.pi  # adimensional time or the phase

# Print outputs for various terms
#print("linear coefficient ", duffing.linear_coefficient.shape, " = \n", duffing.linear_coefficient)
#print("linear term ", duffing.linear_term(state).shape, " = \n", duffing.linear_term(state))
print("nonlinear term ", duffing.nonlinear_term(state, tau).shape, " = \n", duffing.nonlinear_term(state, tau))
#print("jacobian nonlinear_term ", duffing.jacobian_nonlinear_term(state, tau).shape, " = \n", duffing.jacobian_nonlinear_term(state, tau))
#print("external term ", duffing.external_term(tau).shape, " = \n", duffing.external_term(tau))
#print("all terms = \n", duffing.all_terms(state, tau))

# Test with vectorized state
number_of_time_samples = 16
state_vectorized = np.hstack((np.arange(number_of_time_samples).reshape(number_of_time_samples, 1, 1)*0+2, 
                              np.zeros((number_of_time_samples,)).reshape(number_of_time_samples, 1, 1))) # some sequence of states

tau_vectorized = np.linspace(0, 2*np.pi, number_of_time_samples, endpoint=False)

#print("sequence of states", state_vectorized.shape)
print("sequence of nonlinear terms", duffing.nonlinear_term(state_vectorized, tau_vectorized).shape)
#print("sequence of external terms", duffing.external_term(tau_vectorized).shape)
print("sequence of jacobians of the nonlinear terms", duffing.jacobian_nonlinear_term(state_vectorized, tau_vectorized).shape, "\n", 
      np.round(duffing.jacobian_nonlinear_term(state_vectorized, tau_vectorized),9))"""