import numpy as np
    
class ElementStEPConservative(object):
    def __init__(self, c: float = 0, order: int = 3):
        """
        Initializes the ArchBeamSSM parameters.

        :param P: Amplitude multiplier of the external force
        """

        self.omega0 = 3231.28305

        self.linear_coefficient = np.array([
            [-c, -self.omega0],
            [self.omega0, -c]
        ])
        
        self.dimension = self.linear_coefficient.shape[0] # 1 dimensional in second order and 2 dimensional in first order
        self.polynomial_degree = order

    def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
        """
        Calculates the external forcing term.

        :param adimensional_time: Time at which to evaluate the external force
        :return: External force array
        """
        zero = np.zeros_like(adimensional_time)
        return np.array([[zero, zero]]).transpose()
    
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
        x0 = state[..., 0:1, :]
        x1 = state[..., 1:2, :]

        # 26 terms # 29 terms
        fnl = \
            (-2.09514652j) * (x0**3) +\
            ((2.09514652+0j)) * (x0**2) * x1 +\
            (-2.09514652j) * x0 * (x1**2) +\
            ((2.09514652+0j)) * (x1**3) 
        """+\
            (0.00355744829j) * (x0**5) +\
            ((-0.0035574482899999992+0j)) * (x0**4) * x1 +\
            (0.0071148965799999984j) * (x0**3) * (x1**2) +\
            ((-0.007114896579999999+0j)) * (x0**2) * (x1**3) +\
            (0.0035574482899999992j) * x0 * (x1**4) +\
            ((-0.00355744829+0j)) * (x1**5)"""
        
        return -np.concatenate((fnl.real, fnl.imag), axis=-2)
    
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
        x0 = state[..., 0:1, :]
        x1 = state[..., 1:2, :]

        dx0 = \
            (-2.09514652j) * 3 * (x0**2) +\
            ((2.09514652+0j)) * 2 * x0 * x1 +\
            (-2.09514652j) * (x1**2) 
        """+\
            (0.00355744829j) * 5 * (x0**4) +\
            ((-0.0035574482899999992+0j)) * 4 * (x0**3) * x1 +\
            (0.0071148965799999984j) * 3 * (x0**2) * (x1**2) +\
            ((-0.007114896579999999+0j)) * 2 * x0 * (x1**3) +\
            (0.0035574482899999992j) * (x1**4)"""
            
        dx1 = \
            ((2.09514652+0j)) * (x0**2) +\
            (-2.09514652j) * 2 * x0 * x1 +\
            ((2.09514652+0j)) * 3 * (x1**2)
        """+\
            ((-0.0035574482899999992+0j)) * (x0**4) +\
            (0.0071148965799999984j) * 2 * (x0**3) * x1 +\
            ((-0.007114896579999999+0j)) * 3 * (x0**2) * (x1**2) +\
            (0.0035574482899999992j) * 4 * x0 * (x1**3) +\
            ((-0.00355744829+0j)) * 5 * (x1**4)"""

        jacobian0 = np.concatenate((dx0.real, dx1.real), axis=-1)
        jacobian1 = np.concatenate((dx0.imag, dx1.imag), axis=-1)
        
        return -np.concatenate((jacobian0, jacobian1), axis=-2)