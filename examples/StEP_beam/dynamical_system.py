import numpy as np

class DynamicalSystem:
  """
  Base class for dynamical systems.
  Class that implements the dynamics
  zdot = omega z' = f(z, tau) 
  tau = omega * t
  
  """
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
    
class BeamStEP(DynamicalSystem):
    def __init__(self, P: float = 1.0, c: float = 0.0):
        """
        Initializes the ArchBeamSSM parameters.

        :param P: Amplitude multiplier of the external force
        """
        self.P = P

        self.k = 712.896944
        self.c = 7.12896944e-07 + c

        self.linear_coefficient = np.array(
            [[-self.c, -self.k],
             [self.k, -self.c]]
        )
        
        self.dimension = self.linear_coefficient.shape[0] # 1 dimensional in second order and 2 dimensional in first order
        self.polynomial_degree = 7

        self.reference_force = P
        self.reference_length = P/self.k
        self.reference_energy = self.reference_force * self.reference_length

    def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
        """
        Calculates the external forcing term.

        :param adimensional_time: Time at which to evaluate the external force
        :return: External force array
        """
        x1 = self.P * np.cos(adimensional_time)
        x3 = self.P * np.sin(adimensional_time)

        fext =  \
            (-3.50435868e-16-7.01363646e-07j) * x1 +\
            (7.01363646e-07-3.50435868e-16j) * x3

        return np.array([[fext.real, fext.imag]]).transpose()
    
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
        x2 = state[..., 1:2, :]

        x1 = self.P * np.array(np.cos(adimensional_time))[...,np.newaxis,np.newaxis]
        x3 = self.P * np.array(np.sin(adimensional_time))[...,np.newaxis,np.newaxis]

        # 26 terms # 29 terms
        fnl = \
            (1.224554431043e-19-2.3346397949999998e-07j) * (x0**2) +\
            (1.316600000006104e-12-7.09039142e-19j) * x1 * x2 +\
            (-1.225239108957e-19+2.4791499999999934e-11j) * (x2**2) +\
            (4.1359030627651384e-25+24324.2033j) * (x0**3) +\
            (3.6252424380000005e-05j) * (x0**2) * x1 +\
            (1.3999393559999999e-08j) * (x0**2) * x2 +\
            (4.1359030627651384e-25-3.4285433340191533e-06j) * x0 * (x1**2) +\
            (2.0659279999999904e-11j) * x0 * x1 * x3 +\
            (3.0304200000027777j) * x0 * (x2**2) +\
            (-2.985159999999333e-09j) * x0 * x2 * x3 +\
            (4.1359030627651384e-25+3.4285433539808467e-06j) * x0 * (x3**2) +\
            (-9.254049921356641e-09j) * (x1**2) * x2 +\
            (3.521717799999978e-07j) * x1 * (x2**2) +\
            (1.8743637199999997e-09j) * (x2**3) +\
            (9.254049918643358e-09j) * x2 * (x3**2) +\
            (-2.0194839173657902e-28-3.5635504119999994e-07j) * (x0**3) * x1 +\
            (6.925576430000007e-12+3.827274309e-08j) * (x0**2) * (x1*+2) +\
            (1.3234889800848443e-23+1.9782490607999998e-11j) * (x0**2) * x1 * x2 +\
            (9.235314088000001e-11j) * (x0**2) * x1 * x3 +\
            (4.0389678347315804e-28+5.191999999802176e-13j) * (x0**2) * x2 * x3 +\
            (6.925576429999997e-12-5.600197227e-08j) * (x0**2) * (x3**2) +\
            (-1.4307999999994942e-12+9.233076892e-11j) * x0 * (x1**2) * x2 +\
            (9.319915480000002e-08j) * x0 * x1 * (x2**2) +\
            (-6.755471999999604e-11j) * x0 * x1 * x2 * x3 +\
            (-1.4307999999994942e-12-6.451320851999999e-11j) * x0 * x2 * (x3**2) +\
            (6.92557663e-12-5.227847000000099e-11j) * (x1**2) * (x2**2) +\
            (3.308722450212111e-24+6.539975583999999e-12j) * x1 * (x2**3) +\
            (1.3851158319999995e-11j) * x1 * (x2**2) * x3 +\
            (6.92557663e-12-3.47519100000018e-11j) * (x2**2) * (x3**2) 
        
        return np.concatenate((fnl.real, fnl.imag), axis=-2)
    
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
        x2 = state[..., 1:2, :]

        x1 = self.P * np.array(np.cos(adimensional_time))[...,np.newaxis,np.newaxis]
        x3 = self.P * np.array(np.sin(adimensional_time))[...,np.newaxis,np.newaxis]

        dx0 = \
            (1.224554431043e-19-2.3346397949999998e-07j) * 2 * x0 +\
            (4.1359030627651384e-25+24324.2033j) * 3 * (x0**2) +\
            (3.6252424380000005e-05j) * 2 * x0 * x1 +\
            (1.3999393559999999e-08j) * 2 * x0 * x2 +\
            (4.1359030627651384e-25-3.4285433340191533e-06j) * (x1**2) +\
            (2.0659279999999904e-11j) * x1 * x3 +\
            (3.0304200000027777j) * (x2**2) +\
            (-2.985159999999333e-09j) * x2 * x3 +\
            (4.1359030627651384e-25+3.4285433539808467e-06j) * (x3**2) +\
            (-2.0194839173657902e-28-3.5635504119999994e-07j) * 3 * (x0**2) * x1 +\
            (6.925576430000007e-12+3.827274309e-08j) * 2 * x0 * (x1*+2) +\
            (1.3234889800848443e-23+1.9782490607999998e-11j) * 2 * x0 * x1 * x2 +\
            (9.235314088000001e-11j) * 2 * x0 * x1 * x3 +\
            (4.0389678347315804e-28+5.191999999802176e-13j) * 2 * x0 * x2 * x3 +\
            (6.925576429999997e-12-5.600197227e-08j) * 2 * x0 * (x3**2) +\
            (-1.4307999999994942e-12+9.233076892e-11j) * (x1**2) * x2 +\
            (9.319915480000002e-08j) * x1 * (x2**2) +\
            (-6.755471999999604e-11j) * x1 * x2 * x3 +\
            (-1.4307999999994942e-12-6.451320851999999e-11j) * x2 * (x3**2)
            
        dx2 = \
            (1.316600000006104e-12-7.09039142e-19j) * x1 +\
            (-1.225239108957e-19+2.4791499999999934e-11j) * 2 * x2 +\
            (1.3999393559999999e-08j) * (x0**2) +\
            (3.0304200000027777j) * x0 * 2 * x2 +\
            (-2.985159999999333e-09j) * x0 * x3 +\
            (-9.254049921356641e-09j) * (x1**2) +\
            (3.521717799999978e-07j) * x1 * 2 * x2 +\
            (1.8743637199999997e-09j) * 3 * (x2**2) +\
            (9.254049918643358e-09j) * (x3**2) +\
            (1.3234889800848443e-23+1.9782490607999998e-11j) * (x0**2) * x1 +\
            (4.0389678347315804e-28+5.191999999802176e-13j) * (x0**2) * x3 +\
            (-1.4307999999994942e-12+9.233076892e-11j) * x0 * (x1**2) +\
            (9.319915480000002e-08j) * x0 * x1 * 2 * x2 +\
            (-6.755471999999604e-11j) * x0 * x1 * x3 +\
            (-1.4307999999994942e-12-6.451320851999999e-11j) * x0 * (x3**2) +\
            (6.92557663e-12-5.227847000000099e-11j) * (x1**2) * 2 * x2 +\
            (3.308722450212111e-24+6.539975583999999e-12j) * x1 * 3 * (x2**2) +\
            (1.3851158319999995e-11j) * x1 * 2 * x2 * x3 +\
            (6.92557663e-12-3.47519100000018e-11j) * 2 * x2 * (x3**2) 

        jacobian0 = np.concatenate((dx0.real, dx2.real), axis=-1)
        jacobian1 = np.concatenate((dx0.imag, dx2.imag), axis=-1)
        
        return np.concatenate((jacobian0, jacobian1), axis=-2)