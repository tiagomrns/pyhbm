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
    
class ElementStEP(DynamicalSystem):
    def __init__(self, P: float = 1.0, c: float = 1e-3):
        """
        Initializes the ArchBeamSSM parameters.

        :param P: Amplitude multiplier of the external force
        """
        self.P = P

        self.k = 2.61904232
        self.c = c

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

        # 8 terms
        fext =  \
            (-0.190909477j) * x1 +\
            (0.190909477+0j) * x3 +\
            (-6.902531793046327e-31+0.023495341910959998j) * (x1**3) +\
            (0.02807215604999999-1.0963568079999473e-16j) * (x1**2) * x3 +\
            (2.0707595379138983e-30-0.07048521588904j) * x1 * (x3**2) +\
            (-0.009357385349999999-1.0963568080000175e-16j) * (x3**3)

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
            (-0.00223059473j) * (x0**3) +\
            (0.0960127471064j) * (x0**2) * x1 +\
            (0.09547862363291j) * x0 * (x1**2) +\
            (-1.744998199992245e-07+0j) * x0 * x1 * x3 +\
            (0.00427689531j) * x0 * (x2**2) +\
            (-0.0004901391232000032j) * x0 * x2 * x3 +\
            (0.09547505996709j) * x0 * (x3**2) +\
            (-8.72499100030817e-08-5.9904676892e-15j) * (x1**2) * x2 +\
            (0.09561062209360001j) * x1 * (x2**2) +\
            (-1.769635621999977e-05j) * x1 * x2 * x3 +\
            (8.72499100030817e-08+6.0420609508e-15j) * x2 * (x3**2) +\
            (-2.8992661200000004e-17+1.9701068785102994e-08j) * (x0**2) * (x1**2) +\
            (8.758805976147543e-10-1.6414844719999998e-17j) * (x0**2) * x1 * x3 +\
            (-2.899266119999999e-17-1.9701069049724236e-08j) * (x0**2) * (x3**2) +\
            (3.851859888774472e-34+0.0006031988054888648j) * x0 * (x1**3) +\
            (-7.989402065764376e-10-2.527804348e-16j) * x0 * (x1**2) * x2 +\
            (0.001997805268263028+4.196536546e-17j) * x0 * (x1**2) * x3 +\
            (-2.6518525690345542e-08j) * x0 * x1 * x2 * x3 +\
            (-1.5407439555097887e-33-0.0018095860573111352j) * x0 * x1 * (x3**2) +\
            (7.98949948653077e-10+9.39675236e-17j) * x0 * x2 * (x3**2) +\
            (-0.0006659339225369725-1.4663465580000002e-17j) * x0 * (x3**3) +\
            (-0.00037660669113697136-9.5961173e-18j) * (x1**3) * x2 +\
            (-2.89926612e-17+6.441806019295247e-09j) * (x1**2) * (x2**2) +\
            (-1.5407439555097887e-33+0.0028597994311111603j) * (x1**2) * x2 * x3 +\
            (2.473770762385243e-09-6.292479512000001e-17j) * x1 * (x2**2) * x3 +\
            (0.0011298235740630287+4.10494446e-18j) * x1 * x2 * (x3**2) +\
            (-2.899266119999999e-17-6.441806145877525e-09j) * (x2**2) * (x3**2) +\
            (7.703719777548943e-34-0.0009532699300888392j) * x2 * (x3**3)
        
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
            (-0.00223059473j) * 3 * (x0**2) +\
            (0.0960127471064j) * 2 * x0 * x1 +\
            (0.09547862363291j) * (x1**2) +\
            (-1.744998199992245e-07+0j) * x1 * x3 +\
            (0.00427689531j) * (x2**2) +\
            (-0.0004901391232000032j) * x2 * x3 +\
            (0.09547505996709j) * (x3**2) +\
            (-2.8992661200000004e-17+1.9701068785102994e-08j) * 2 * x0 * (x1**2) +\
            (8.758805976147543e-10-1.6414844719999998e-17j) * 2 * x0 * x1 * x3 +\
            (-2.899266119999999e-17-1.9701069049724236e-08j) * 2 * x0 * (x3**2) +\
            (3.851859888774472e-34+0.0006031988054888648j) * (x1**3) +\
            (-7.989402065764376e-10-2.527804348e-16j) * (x1**2) * x2 +\
            (0.001997805268263028+4.196536546e-17j) * (x1**2) * x3 +\
            (-2.6518525690345542e-08j) * x1 * x2 * x3 +\
            (-1.5407439555097887e-33-0.0018095860573111352j) * x1 * (x3**2) +\
            (7.98949948653077e-10+9.39675236e-17j) * x2 * (x3**2) +\
            (-0.0006659339225369725-1.4663465580000002e-17j) * (x3**3)
            
        dx2 = \
            (0.00427689531j) * x0 * 2 * x2 +\
            (-0.0004901391232000032j) * x0 * x3 +\
            (-8.72499100030817e-08-5.9904676892e-15j) * (x1**2) +\
            (0.09561062209360001j) * x1 * 2 * x2 +\
            (-1.769635621999977e-05j) * x1 * x3 +\
            (8.72499100030817e-08+6.0420609508e-15j) * (x3**2) +\
            (-2.899266119999999e-17-6.441806145877525e-09j) * 2 * x2 * (x3**2) +\
            (-7.989402065764376e-10-2.527804348e-16j) * x0 * (x1**2) +\
            (-2.6518525690345542e-08j) * x0 * x1 * x3 +\
            (-1.5407439555097887e-33+0.0028597994311111603j) * (x1**2) * x3 +\
            (7.98949948653077e-10+9.39675236e-17j) * x0 * (x3**2) +\
            (-0.00037660669113697136-9.5961173e-18j) * (x1**3) +\
            (-2.89926612e-17+6.441806019295247e-09j) * (x1**2) * 2 * x2 +\
            (2.473770762385243e-09-6.292479512000001e-17j) * x1 * 2 * x2 * x3 +\
            (0.0011298235740630287+4.10494446e-18j) * x1 * (x3**2) +\
            (7.703719777548943e-34-0.0009532699300888392j) * (x3**3)

        jacobian0 = np.concatenate((dx0.real, dx2.real), axis=-1)
        jacobian1 = np.concatenate((dx0.imag, dx2.imag), axis=-1)
        
        return np.concatenate((jacobian0, jacobian1), axis=-2)

class ElementStEPLinear(DynamicalSystem):
    def __init__(self, P: float = 1.0, c: float = 1e-3):
        """
        Initializes the ArchBeamSSM parameters.

        :param P: Amplitude multiplier of the external force
        """
        self.P = P

        self.k = 2.61904232
        self.c = c

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
            (-0.190909477j) * x1 +\
            (0.190909477+0j) * x3

        return np.array([[fext.real, fext.imag]]).transpose()
    
    def linear_term(self, state: np.ndarray) -> np.ndarray:
        """
        Calculates the linear term.

        :param state: State vector
        :return: Linear term array
        """
        return self.linear_coefficient @ state

    def nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(state[..., 0:1, :])
        return np.concatenate((zeros, zeros), axis=-2)
    
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
        zeros = np.zeros_like(state[..., 0:1, :])
        jacobian1 = np.concatenate((zeros, zeros), axis=-1)
        return np.concatenate((jacobian1, jacobian1), axis=-2)