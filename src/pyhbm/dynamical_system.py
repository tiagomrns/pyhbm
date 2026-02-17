import numpy as np
from numpy.typing import ArrayLike


class FirstOrderODE:
    """
    Base class for dynamical systems.
    
    Class that implements the dynamics:
        zdot = omega * z' = f(z, tau)
        tau = omega * t
        
        -> The class writes the function f(z, tau) = A @ z + fnl(z, tau) + fext(tau)
        The adimensional-time ODE becomes:  z' = f(z, tau) / omega
    
    Subclasses must implement:
        - external_term(adimensional_time), # fext
        - linear_coefficient(state), # A
        - nonlinear_term(state, adimensional_time), # fnl
    """
    
    is_real_valued: bool = True
    
    def __init__(self):
        self.linear_coefficient: np.ndarray = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.dimension: int = 2
        self.polynomial_degree: int = 1
    
    def external_term(self, adimensional_time: ArrayLike) -> np.ndarray:
        """
        Calculate the external forcing term.
        
        :param adimensional_time: Adimensional time at which to evaluate the external force
        :return: External force array of shape (dimension, len(adimensional_time))
        """
        raise NotImplementedError("Subclasses must implement external_term.")
    
    def linear_term(self, state: ArrayLike) -> np.ndarray:
        """
        Calculate the linear term.
        
        :param state: State vector
        :return: Linear term array
        """
        raise NotImplementedError("Subclasses must implement linear_term.")
    
    def nonlinear_term(self, state: ArrayLike, adimensional_time: ArrayLike) -> np.ndarray:
        """
        Calculate the nonlinear term.
        
        :param state: State vector
        :param adimensional_time: Adimensional time
        :return: Nonlinear term array
        """
        raise NotImplementedError("Subclasses must implement nonlinear_term.")
    
    def jacobian_nonlinear_term(self, state: ArrayLike, adimensional_time: ArrayLike) -> np.ndarray:
        """
        Calculate the Jacobian of the nonlinear term.
        
        :param state: State vector
        :param adimensional_time: Adimensional time
        :return: Jacobian array of shape (dimension, dimension, ...)
        """
        raise NotImplementedError("Subclasses must implement jacobian_nonlinear_term.")
    
    def all_terms(self, state: ArrayLike, adimensional_time: ArrayLike) -> np.ndarray:
        """
        Combine all terms (linear, nonlinear, external) to compute the total force.
        
        :param state: State vector
        :param adimensional_time: Adimensional time
        :return: Total force array
        """
        return (
            self.linear_term(state)
            + self.nonlinear_term(state, adimensional_time)
            + self.external_term(adimensional_time)
        )

    def compute_jacobian(self, state: ArrayLike, adimensional_time: ArrayLike) -> np.ndarray:
        """
        Compute the Jacobian of the full dynamical system with respect to state.
        
        :param state: State vector
        :param adimensional_time: Adimensional time
        :return: Jacobian array of shape (dimension, dimension)
        """
        return self.linear_coefficient + self.jacobian_nonlinear_term(state, adimensional_time)
