#%% non-autonomous time-normalized state-space function dx/dt = f_new(adimensional_time, x)
import numpy as np
from numpy import cos, sin, array, concatenate

class ArchBeamSSM(object):
  
  is_real_valued = True
  
  def __init__(self, P: float = 1.0):
    """
    Initializes the ArchBeamSSM parameters.

    :param P: Amplitude multiplier of the external force
    """
    self.P = P

    self.omega0 = 1.03308419e+00
    self.c = 1.03308471e-03

    self.linear_coefficient = array(
       [[-self.c, self.omega0],
        [-self.omega0, -self.c]]
    )
    
    self.dimension = self.linear_coefficient.shape[0] # 1 dimensional in second order and 2 dimensional in first order
    self.polynomial_degree = 7

  def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
    """
    Calculates the external forcing term.

    :param adimensional_time: Time at which to evaluate the external force
    :return: External force array
    """
    Psin = self.P * sin(adimensional_time)
    Pcos = self.P * cos(adimensional_time)

    f0 = \
        +1.45196300e-02*Psin+\
        -1.72247591e-14*Pcos+\
        -9.55361860e-04*Psin**3+\
        +5.62122730e-07*Psin**2*Pcos+\
        -9.55361860e-04*Psin*Pcos**2+\
        +5.62122730e-07*Pcos**3
    
    f1 = \
        +1.72247591e-14*Psin+\
        +1.45196300e-02*Pcos+\
        -5.62122730e-07*Psin**3+\
        -9.55361860e-04*Psin**2*Pcos+\
        -5.62122730e-07*Psin*Pcos**2+\
        -9.55361860e-04*Pcos**3

    return array([[f0, f1]]).transpose()
    
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

    Psin = self.P * array(sin(adimensional_time))[...,np.newaxis,np.newaxis]
    Pcos = self.P * array(cos(adimensional_time))[...,np.newaxis,np.newaxis]

    f0 = \
        +1.58511993e-08*x0**3+\
        -2.70048831e-05*x0**2*x1+\
        -7.62998410e-05*x0**2*Psin+\
        -1.93314336e-08*x0**2*Pcos+\
        +1.58511993e-08*x0*x1**2+\
        +7.05327065e-08*x0*x1*Psin+\
        +1.51708814e-04*x0*x1*Pcos+\
        +1.73485441e-07*x0*Psin**2+\
        -3.80139338e-04*x0*Psin*Pcos+\
        -1.77948983e-07*x0*Pcos**2+\
        -2.70048831e-05*x1**3+\
        -2.28008655e-04*x1**2*Psin+\
        +5.12012730e-08*x1**2*Pcos+\
        -4.09380791e-04*x1*Psin**2+\
        +3.51434424e-07*x1*Psin*Pcos+\
        -7.89520129e-04*x1*Pcos**2+\
        -3.11635113e-12*x0**5+\
        +5.30068134e-09*x0**4*x1+\
        +1.12933884e-08*x0**4*Psin+\
        +1.24119784e-11*x0**4*Pcos+\
        -6.23270226e-12*x0**3*x1**2+\
        -1.09479997e-11*x0**3*x1*Psin+\
        -4.30961361e-08*x0**3*x1*Pcos+\
        -2.74144918e-11*x0**3*Psin**2+\
        -9.18570279e-08*x0**3*Psin*Pcos+\
        -1.02298543e-10*x0**3*Pcos**2+\
        +1.06013627e-08*x0**2*x1**3+\
        +6.56829129e-08*x0**2*x1**2*Psin+\
        +1.38759571e-11*x0**2*x1**2*Pcos+\
        +3.73284881e-07*x0**2*x1*Psin**2+\
        +3.83341741e-10*x0**2*x1*Psin*Pcos+\
        +3.92658453e-07*x0**2*x1*Pcos**2+\
        +7.24932022e-07*x0**2*Psin**3+\
        +7.14441549e-10*x0**2*Psin**2*Pcos+\
        +9.19974098e-07*x0**2*Psin*Pcos**2+\
        -4.95342895e-11*x0**2*Pcos**3+\
        -3.11635113e-12*x0*x1**4+\
        -1.09479997e-11*x0*x1**3*Psin+\
        -4.30961361e-08*x0*x1**3*Pcos+\
        -3.35872182e-10*x0*x1**2*Psin**2+\
        -3.14318229e-07*x0*x1**2*Psin*Pcos+\
        +2.06159148e-10*x0*x1**2*Pcos**2+\
        -1.00443071e-09*x0*x1*Psin**3+\
        -1.86621602e-06*x0*x1*Psin**2*Pcos+\
        +5.23520966e-10*x0*x1*Psin*Pcos**2+\
        -1.47613187e-06*x0*x1*Pcos**3+\
        +5.30068134e-09*x1**5+\
        +5.43895245e-08*x1**4*Psin+\
        +1.46397868e-12*x1**4*Pcos+\
        +4.84515481e-07*x1**3*Psin**2+\
        -2.33573640e-10*x1**3*Psin*Pcos+\
        +2.81427853e-07*x1**3*Pcos**2+\
        +2.39610597e-06*x1**2*Psin**3+\
        -1.05396500e-09*x1**2*Psin**2*Pcos+\
        +2.20106389e-06*x1**2*Psin*Pcos**2+\
        -2.89989162e-10*x1**2*Pcos**3+\
        -1.23914182e-15*x0**7+\
        +6.89071790e-13*x0**6*x1+\
        +1.44173163e-12*x0**6*Psin+\
        +1.66578576e-14*x0**6*Pcos+\
        -3.71742547e-15*x0**5*x1**2+\
        -1.97807743e-14*x0**5*x1*Psin+\
        -9.37110256e-12*x0**5*x1*Pcos+\
        -5.99146934e-14*x0**5*Psin**2+\
        -1.98706199e-11*x0**5*Psin*Pcos+\
        -8.79705113e-14*x0**5*Pcos**2+\
        +2.06721537e-12*x0**4*x1**3+\
        +1.36962975e-11*x0**4*x1**2*Psin+\
        +3.01927987e-14*x0**4*x1**2*Pcos+\
        +5.12070188e-11*x0**4*x1*Psin**2+\
        +6.97301291e-14*x0**4*x1*Psin*Pcos+\
        +5.69058250e-11*x0**4*x1*Pcos**2+\
        +6.25777798e-11*x0**4*Psin**3+\
        +3.26695118e-13*x0**4*Psin**2*Pcos+\
        +9.31557039e-11*x0**4*Psin*Pcos**2+\
        +2.97065567e-13*x0**4*Pcos**3+\
        -3.71742547e-15*x0**3*x1**4+\
        -3.95615485e-14*x0**3*x1**3*Psin+\
        -1.87422051e-11*x0**3*x1**3*Pcos+\
        -1.61503698e-13*x0**3*x1**2*Psin**2+\
        -9.08800920e-11*x0**3*x1**2*Psin*Pcos+\
        -1.34266711e-13*x0**3*x1**2*Pcos**2+\
        -5.35213233e-13*x0**3*x1*Psin**3+\
        -2.85124332e-10*x0**3*x1*Psin**2*Pcos+\
        -4.16904430e-13*x0**3*x1*Psin*Pcos**2+\
        -1.98148081e-10*x0**3*x1*Pcos**3+\
        +2.06721537e-12*x0**2*x1**5+\
        +2.30674000e-11*x0**2*x1**4*Psin+\
        +1.04120244e-14*x0**2*x1**4*Pcos+\
        +1.27983464e-10*x0**2*x1**3*Psin**2+\
        +5.61116357e-14*x0**2*x1**3*Psin*Pcos+\
        +8.82422239e-11*x0**2*x1**3*Pcos**2+\
        +3.79701968e-10*x0**2*x1**2*Psin**3+\
        -2.95519494e-14*x0**2*x1**2*Psin**2*Pcos+\
        +3.02240758e-10*x0**2*x1**2*Psin*Pcos**2+\
        +1.47597153e-13*x0**2*x1**2*Pcos**3+\
        -1.23914182e-15*x0*x1**6+\
        -1.97807743e-14*x0*x1**5*Psin+\
        -9.37110256e-12*x0*x1**5*Pcos+\
        -1.01589005e-13*x0*x1**4*Psin**2+\
        -7.10094721e-11*x0*x1**4*Psin*Pcos+\
        -4.62962000e-14*x0*x1**4*Pcos**2+\
        -4.76163532e-13*x0*x1**3*Psin**3+\
        -2.07663122e-10*x0*x1**3*Psin**2*Pcos+\
        -5.94053532e-13*x0*x1**3*Psin*Pcos**2+\
        -2.23968484e-10*x0*x1**3*Pcos**3+\
        +6.89071790e-13*x1**7+\
        +1.08128342e-11*x1**6*Psin+\
        -3.12291661e-15*x1**6*Pcos+\
        +7.67764449e-11*x1**5*Psin**2+\
        -1.36184934e-14*x1**5*Psin*Pcos+\
        +3.13363989e-11*x1**5*Pcos**2+\
        +2.91303785e-10*x1**4*Psin**3+\
        -1.79097965e-13*x1**4*Psin**2*Pcos+\
        +2.86546264e-10*x1**4*Psin*Pcos**2+\
        -2.08518115e-13*x1**4*Pcos**3
    
    f1 = \
        +2.70048831e-05*x0**3+\
        +1.58511993e-08*x0**2*x1+\
        -5.12012730e-08*x0**2*Psin+\
        -2.28008655e-04*x0**2*Pcos+\
        +2.70048831e-05*x0*x1**2+\
        +1.51708814e-04*x0*x1*Psin+\
        -7.05327065e-08*x0*x1*Pcos+\
        +7.89520129e-04*x0*Psin**2+\
        +3.51434424e-07*x0*Psin*Pcos+\
        +4.09380791e-04*x0*Pcos**2+\
        +1.58511993e-08*x1**3+\
        +1.93314336e-08*x1**2*Psin+\
        -7.62998410e-05*x1**2*Pcos+\
        -1.77948983e-07*x1*Psin**2+\
        +3.80139338e-04*x1*Psin*Pcos+\
        +1.73485441e-07*x1*Pcos**2+\
        -5.30068134e-09*x0**5+\
        -3.11635113e-12*x0**4*x1+\
        -1.46397868e-12*x0**4*Psin+\
        +5.43895245e-08*x0**4*Pcos+\
        -1.06013627e-08*x0**3*x1**2+\
        -4.30961361e-08*x0**3*x1*Psin+\
        +1.09479997e-11*x0**3*x1*Pcos+\
        -2.81427853e-07*x0**3*Psin**2+\
        -2.33573640e-10*x0**3*Psin*Pcos+\
        -4.84515481e-07*x0**3*Pcos**2+\
        -6.23270226e-12*x0**2*x1**3+\
        -1.38759571e-11*x0**2*x1**2*Psin+\
        +6.56829129e-08*x0**2*x1**2*Pcos+\
        +2.06159148e-10*x0**2*x1*Psin**2+\
        +3.14318229e-07*x0**2*x1*Psin*Pcos+\
        -3.35872182e-10*x0**2*x1*Pcos**2+\
        +2.89989162e-10*x0**2*Psin**3+\
        +2.20106389e-06*x0**2*Psin**2*Pcos+\
        +1.05396500e-09*x0**2*Psin*Pcos**2+\
        +2.39610597e-06*x0**2*Pcos**3+\
        -5.30068134e-09*x0*x1**4+\
        -4.30961361e-08*x0*x1**3*Psin+\
        +1.09479997e-11*x0*x1**3*Pcos+\
        -3.92658453e-07*x0*x1**2*Psin**2+\
        +3.83341741e-10*x0*x1**2*Psin*Pcos+\
        -3.73284881e-07*x0*x1**2*Pcos**2+\
        -1.47613187e-06*x0*x1*Psin**3+\
        -5.23520966e-10*x0*x1*Psin**2*Pcos+\
        -1.86621602e-06*x0*x1*Psin*Pcos**2+\
        +1.00443071e-09*x0*x1*Pcos**3+\
        -3.11635113e-12*x1**5+\
        -1.24119784e-11*x1**4*Psin+\
        +1.12933884e-08*x1**4*Pcos+\
        -1.02298543e-10*x1**3*Psin**2+\
        +9.18570279e-08*x1**3*Psin*Pcos+\
        -2.74144918e-11*x1**3*Pcos**2+\
        +4.95342895e-11*x1**2*Psin**3+\
        +9.19974098e-07*x1**2*Psin**2*Pcos+\
        -7.14441549e-10*x1**2*Psin*Pcos**2+\
        +7.24932022e-07*x1**2*Pcos**3+\
        -6.89071790e-13*x0**7+\
        -1.23914182e-15*x0**6*x1+\
        +3.12291661e-15*x0**6*Psin+\
        +1.08128342e-11*x0**6*Pcos+\
        -2.06721537e-12*x0**5*x1**2+\
        -9.37110256e-12*x0**5*x1*Psin+\
        +1.97807743e-14*x0**5*x1*Pcos+\
        -3.13363989e-11*x0**5*Psin**2+\
        -1.36184934e-14*x0**5*Psin*Pcos+\
        -7.67764449e-11*x0**5*Pcos**2+\
        -3.71742547e-15*x0**4*x1**3+\
        -1.04120244e-14*x0**4*x1**2*Psin+\
        +2.30674000e-11*x0**4*x1**2*Pcos+\
        -4.62962000e-14*x0**4*x1*Psin**2+\
        +7.10094721e-11*x0**4*x1*Psin*Pcos+\
        -1.01589005e-13*x0**4*x1*Pcos**2+\
        +2.08518115e-13*x0**4*Psin**3+\
        +2.86546264e-10*x0**4*Psin**2*Pcos+\
        +1.79097965e-13*x0**4*Psin*Pcos**2+\
        +2.91303785e-10*x0**4*Pcos**3+\
        -2.06721537e-12*x0**3*x1**4+\
        -1.87422051e-11*x0**3*x1**3*Psin+\
        +3.95615485e-14*x0**3*x1**3*Pcos+\
        -8.82422239e-11*x0**3*x1**2*Psin**2+\
        +5.61116357e-14*x0**3*x1**2*Psin*Pcos+\
        -1.27983464e-10*x0**3*x1**2*Pcos**2+\
        -2.23968484e-10*x0**3*x1*Psin**3+\
        +5.94053532e-13*x0**3*x1*Psin**2*Pcos+\
        -2.07663122e-10*x0**3*x1*Psin*Pcos**2+\
        +4.76163532e-13*x0**3*x1*Pcos**3+\
        -3.71742547e-15*x0**2*x1**5+\
        -3.01927987e-14*x0**2*x1**4*Psin+\
        +1.36962975e-11*x0**2*x1**4*Pcos+\
        -1.34266711e-13*x0**2*x1**3*Psin**2+\
        +9.08800920e-11*x0**2*x1**3*Psin*Pcos+\
        -1.61503698e-13*x0**2*x1**3*Pcos**2+\
        -1.47597153e-13*x0**2*x1**2*Psin**3+\
        +3.02240758e-10*x0**2*x1**2*Psin**2*Pcos+\
        +2.95519494e-14*x0**2*x1**2*Psin*Pcos**2+\
        +3.79701968e-10*x0**2*x1**2*Pcos**3+\
        -6.89071790e-13*x0*x1**6+\
        -9.37110256e-12*x0*x1**5*Psin+\
        +1.97807743e-14*x0*x1**5*Pcos+\
        -5.69058250e-11*x0*x1**4*Psin**2+\
        +6.97301291e-14*x0*x1**4*Psin*Pcos+\
        -5.12070188e-11*x0*x1**4*Pcos**2+\
        -1.98148081e-10*x0*x1**3*Psin**3+\
        +4.16904430e-13*x0*x1**3*Psin**2*Pcos+\
        -2.85124332e-10*x0*x1**3*Psin*Pcos**2+\
        +5.35213233e-13*x0*x1**3*Pcos**3+\
        -1.23914182e-15*x1**7+\
        -1.66578576e-14*x1**6*Psin+\
        +1.44173163e-12*x1**6*Pcos+\
        -8.79705113e-14*x1**5*Psin**2+\
        +1.98706199e-11*x1**5*Psin*Pcos+\
        -5.99146934e-14*x1**5*Pcos**2+\
        -2.97065567e-13*x1**4*Psin**3+\
        +9.31557039e-11*x1**4*Psin**2*Pcos+\
        -3.26695118e-13*x1**4*Psin*Pcos**2+\
        +6.25777798e-11*x1**4*Pcos**3
    
    return concatenate((f0, f1), axis=-2)
    
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

    Psin = self.P * array(sin(adimensional_time))[...,np.newaxis,np.newaxis]
    Pcos = self.P * array(cos(adimensional_time))[...,np.newaxis,np.newaxis]

    j00 = \
        +1.58511993e-08*3*x0**2+\
        -2.70048831e-05*2*x0*x1+\
        -7.62998410e-05*2*x0*Psin+\
        -1.93314336e-08*2*x0*Pcos+\
        +1.58511993e-08*1*x1**2+\
        +7.05327065e-08*1*x1*Psin+\
        +1.51708814e-04*1*x1*Pcos+\
        +1.73485441e-07*1*Psin**2+\
        -3.80139338e-04*1*Psin*Pcos+\
        -1.77948983e-07*1*Pcos**2+\
        -3.11635113e-12*5*x0**4+\
        +5.30068134e-09*4*x0**3*x1+\
        +1.12933884e-08*4*x0**3*Psin+\
        +1.24119784e-11*4*x0**3*Pcos+\
        -6.23270226e-12*3*x0**2*x1**2+\
        -1.09479997e-11*3*x0**2*x1*Psin+\
        -4.30961361e-08*3*x0**2*x1*Pcos+\
        -2.74144918e-11*3*x0**2*Psin**2+\
        -9.18570279e-08*3*x0**2*Psin*Pcos+\
        -1.02298543e-10*3*x0**2*Pcos**2+\
        +1.06013627e-08*2*x0*x1**3+\
        +6.56829129e-08*2*x0*x1**2*Psin+\
        +1.38759571e-11*2*x0*x1**2*Pcos+\
        +3.73284881e-07*2*x0*x1*Psin**2+\
        +3.83341741e-10*2*x0*x1*Psin*Pcos+\
        +3.92658453e-07*2*x0*x1*Pcos**2+\
        +7.24932022e-07*2*x0*Psin**3+\
        +7.14441549e-10*2*x0*Psin**2*Pcos+\
        +9.19974098e-07*2*x0*Psin*Pcos**2+\
        -4.95342895e-11*2*x0*Pcos**3+\
        -3.11635113e-12*1*x1**4+\
        -1.09479997e-11*1*x1**3*Psin+\
        -4.30961361e-08*1*x1**3*Pcos+\
        -3.35872182e-10*1*x1**2*Psin**2+\
        -3.14318229e-07*1*x1**2*Psin*Pcos+\
        +2.06159148e-10*1*x1**2*Pcos**2+\
        -1.00443071e-09*1*x1*Psin**3+\
        -1.86621602e-06*1*x1*Psin**2*Pcos+\
        +5.23520966e-10*1*x1*Psin*Pcos**2+\
        -1.47613187e-06*1*x1*Pcos**3+\
        -1.23914182e-15*7*x0**6+\
        +6.89071790e-13*6*x0**5*x1+\
        +1.44173163e-12*6*x0**5*Psin+\
        +1.66578576e-14*6*x0**5*Pcos+\
        -3.71742547e-15*5*x0**4*x1**2+\
        -1.97807743e-14*5*x0**4*x1*Psin+\
        -9.37110256e-12*5*x0**4*x1*Pcos+\
        -5.99146934e-14*5*x0**4*Psin**2+\
        -1.98706199e-11*5*x0**4*Psin*Pcos+\
        -8.79705113e-14*5*x0**4*Pcos**2+\
        +2.06721537e-12*4*x0**3*x1**3+\
        +1.36962975e-11*4*x0**3*x1**2*Psin+\
        +3.01927987e-14*4*x0**3*x1**2*Pcos+\
        +5.12070188e-11*4*x0**3*x1*Psin**2+\
        +6.97301291e-14*4*x0**3*x1*Psin*Pcos+\
        +5.69058250e-11*4*x0**3*x1*Pcos**2+\
        +6.25777798e-11*4*x0**3*Psin**3+\
        +3.26695118e-13*4*x0**3*Psin**2*Pcos+\
        +9.31557039e-11*4*x0**3*Psin*Pcos**2+\
        +2.97065567e-13*4*x0**3*Pcos**3+\
        -3.71742547e-15*3*x0**2*x1**4+\
        -3.95615485e-14*3*x0**2*x1**3*Psin+\
        -1.87422051e-11*3*x0**2*x1**3*Pcos+\
        -1.61503698e-13*3*x0**2*x1**2*Psin**2+\
        -9.08800920e-11*3*x0**2*x1**2*Psin*Pcos+\
        -1.34266711e-13*3*x0**2*x1**2*Pcos**2+\
        -5.35213233e-13*3*x0**2*x1*Psin**3+\
        -2.85124332e-10*3*x0**2*x1*Psin**2*Pcos+\
        -4.16904430e-13*3*x0**2*x1*Psin*Pcos**2+\
        -1.98148081e-10*3*x0**2*x1*Pcos**3+\
        +2.06721537e-12*2*x0*x1**5+\
        +2.30674000e-11*2*x0*x1**4*Psin+\
        +1.04120244e-14*2*x0*x1**4*Pcos+\
        +1.27983464e-10*2*x0*x1**3*Psin**2+\
        +5.61116357e-14*2*x0*x1**3*Psin*Pcos+\
        +8.82422239e-11*2*x0*x1**3*Pcos**2+\
        +3.79701968e-10*2*x0*x1**2*Psin**3+\
        -2.95519494e-14*2*x0*x1**2*Psin**2*Pcos+\
        +3.02240758e-10*2*x0*x1**2*Psin*Pcos**2+\
        +1.47597153e-13*2*x0*x1**2*Pcos**3+\
        -1.23914182e-15*1*x1**6+\
        -1.97807743e-14*1*x1**5*Psin+\
        -9.37110256e-12*1*x1**5*Pcos+\
        -1.01589005e-13*1*x1**4*Psin**2+\
        -7.10094721e-11*1*x1**4*Psin*Pcos+\
        -4.62962000e-14*1*x1**4*Pcos**2+\
        -4.76163532e-13*1*x1**3*Psin**3+\
        -2.07663122e-10*1*x1**3*Psin**2*Pcos+\
        -5.94053532e-13*1*x1**3*Psin*Pcos**2+\
        -2.23968484e-10*1*x1**3*Pcos**3

    j01 = \
        -2.70048831e-05*1*x0**2+\
        +1.58511993e-08*2*x1*x0+\
        +7.05327065e-08*1*x0*Psin+\
        +1.51708814e-04*1*x0*Pcos+\
        -2.70048831e-05*3*x1**2+\
        -2.28008655e-04*2*x1*Psin+\
        +5.12012730e-08*2*x1*Pcos+\
        -4.09380791e-04*1*Psin**2+\
        +3.51434424e-07*1*Psin*Pcos+\
        -7.89520129e-04*1*Pcos**2+\
        +5.30068134e-09*1*x0**4+\
        -6.23270226e-12*2*x1*x0**3+\
        -1.09479997e-11*1*x0**3*Psin+\
        -4.30961361e-08*1*x0**3*Pcos+\
        +1.06013627e-08*3*x1**2*x0**2+\
        +6.56829129e-08*2*x1*x0**2*Psin+\
        +1.38759571e-11*2*x1*x0**2*Pcos+\
        +3.73284881e-07*1*x0**2*Psin**2+\
        +3.83341741e-10*1*x0**2*Psin*Pcos+\
        +3.92658453e-07*1*x0**2*Pcos**2+\
        -3.11635113e-12*4*x1**3*x0+\
        -1.09479997e-11*3*x1**2*x0*Psin+\
        -4.30961361e-08*3*x1**2*x0*Pcos+\
        -3.35872182e-10*2*x1*x0*Psin**2+\
        -3.14318229e-07*2*x1*x0*Psin*Pcos+\
        +2.06159148e-10*2*x1*x0*Pcos**2+\
        -1.00443071e-09*1*x0*Psin**3+\
        -1.86621602e-06*1*x0*Psin**2*Pcos+\
        +5.23520966e-10*1*x0*Psin*Pcos**2+\
        -1.47613187e-06*1*x0*Pcos**3+\
        +5.30068134e-09*5*x1**4+\
        +5.43895245e-08*4*x1**3*Psin+\
        +1.46397868e-12*4*x1**3*Pcos+\
        +4.84515481e-07*3*x1**2*Psin**2+\
        -2.33573640e-10*3*x1**2*Psin*Pcos+\
        +2.81427853e-07*3*x1**2*Pcos**2+\
        +2.39610597e-06*2*x1*Psin**3+\
        -1.05396500e-09*2*x1*Psin**2*Pcos+\
        +2.20106389e-06*2*x1*Psin*Pcos**2+\
        -2.89989162e-10*2*x1*Pcos**3+\
        +6.89071790e-13*1*x0**6+\
        -3.71742547e-15*2*x1*x0**5+\
        -1.97807743e-14*1*x0**5*Psin+\
        -9.37110256e-12*1*x0**5*Pcos+\
        +2.06721537e-12*3*x1**2*x0**4+\
        +1.36962975e-11*2*x1*x0**4*Psin+\
        +3.01927987e-14*2*x1*x0**4*Pcos+\
        +5.12070188e-11*1*x0**4*Psin**2+\
        +6.97301291e-14*1*x0**4*Psin*Pcos+\
        +5.69058250e-11*1*x0**4*Pcos**2+\
        -3.71742547e-15*4*x1**3*x0**3+\
        -3.95615485e-14*3*x1**2*x0**3*Psin+\
        -1.87422051e-11*3*x1**2*x0**3*Pcos+\
        -1.61503698e-13*2*x1*x0**3*Psin**2+\
        -9.08800920e-11*2*x1*x0**3*Psin*Pcos+\
        -1.34266711e-13*2*x1*x0**3*Pcos**2+\
        -5.35213233e-13*1*x0**3*Psin**3+\
        -2.85124332e-10*1*x0**3*Psin**2*Pcos+\
        -4.16904430e-13*1*x0**3*Psin*Pcos**2+\
        -1.98148081e-10*1*x0**3*Pcos**3+\
        +2.06721537e-12*5*x1**4*x0**2+\
        +2.30674000e-11*4*x1**3*x0**2*Psin+\
        +1.04120244e-14*4*x1**3*x0**2*Pcos+\
        +1.27983464e-10*3*x1**2*x0**2*Psin**2+\
        +5.61116357e-14*3*x1**2*x0**2*Psin*Pcos+\
        +8.82422239e-11*3*x1**2*x0**2*Pcos**2+\
        +3.79701968e-10*2*x1*x0**2*Psin**3+\
        -2.95519494e-14*2*x1*x0**2*Psin**2*Pcos+\
        +3.02240758e-10*2*x1*x0**2*Psin*Pcos**2+\
        +1.47597153e-13*2*x1*x0**2*Pcos**3+\
        -1.23914182e-15*6*x1**5*x0+\
        -1.97807743e-14*5*x1**4*x0*Psin+\
        -9.37110256e-12*5*x1**4*x0*Pcos+\
        -1.01589005e-13*4*x1**3*x0*Psin**2+\
        -7.10094721e-11*4*x1**3*x0*Psin*Pcos+\
        -4.62962000e-14*4*x1**3*x0*Pcos**2+\
        -4.76163532e-13*3*x1**2*x0*Psin**3+\
        -2.07663122e-10*3*x1**2*x0*Psin**2*Pcos+\
        -5.94053532e-13*3*x1**2*x0*Psin*Pcos**2+\
        -2.23968484e-10*3*x1**2*x0*Pcos**3+\
        +6.89071790e-13*7*x1**6+\
        +1.08128342e-11*6*x1**5*Psin+\
        -3.12291661e-15*6*x1**5*Pcos+\
        +7.67764449e-11*5*x1**4*Psin**2+\
        -1.36184934e-14*5*x1**4*Psin*Pcos+\
        +3.13363989e-11*5*x1**4*Pcos**2+\
        +2.91303785e-10*4*x1**3*Psin**3+\
        -1.79097965e-13*4*x1**3*Psin**2*Pcos+\
        +2.86546264e-10*4*x1**3*Psin*Pcos**2+\
        -2.08518115e-13*4*x1**3*Pcos**3


    j10 = \
        +2.70048831e-05*3*x0**2+\
        +1.58511993e-08*2*x0*x1+\
        -5.12012730e-08*2*x0*Psin+\
        -2.28008655e-04*2*x0*Pcos+\
        +2.70048831e-05*1*x1**2+\
        +1.51708814e-04*1*x1*Psin+\
        -7.05327065e-08*1*x1*Pcos+\
        +7.89520129e-04*1*Psin**2+\
        +3.51434424e-07*1*Psin*Pcos+\
        +4.09380791e-04*1*Pcos**2+\
        -5.30068134e-09*5*x0**4+\
        -3.11635113e-12*4*x0**3*x1+\
        -1.46397868e-12*4*x0**3*Psin+\
        +5.43895245e-08*4*x0**3*Pcos+\
        -1.06013627e-08*3*x0**2*x1**2+\
        -4.30961361e-08*3*x0**2*x1*Psin+\
        +1.09479997e-11*3*x0**2*x1*Pcos+\
        -2.81427853e-07*3*x0**2*Psin**2+\
        -2.33573640e-10*3*x0**2*Psin*Pcos+\
        -4.84515481e-07*3*x0**2*Pcos**2+\
        -6.23270226e-12*2*x0*x1**3+\
        -1.38759571e-11*2*x0*x1**2*Psin+\
        +6.56829129e-08*2*x0*x1**2*Pcos+\
        +2.06159148e-10*2*x0*x1*Psin**2+\
        +3.14318229e-07*2*x0*x1*Psin*Pcos+\
        -3.35872182e-10*2*x0*x1*Pcos**2+\
        +2.89989162e-10*2*x0*Psin**3+\
        +2.20106389e-06*2*x0*Psin**2*Pcos+\
        +1.05396500e-09*2*x0*Psin*Pcos**2+\
        +2.39610597e-06*2*x0*Pcos**3+\
        -5.30068134e-09*1*x1**4+\
        -4.30961361e-08*1*x1**3*Psin+\
        +1.09479997e-11*1*x1**3*Pcos+\
        -3.92658453e-07*1*x1**2*Psin**2+\
        +3.83341741e-10*1*x1**2*Psin*Pcos+\
        -3.73284881e-07*1*x1**2*Pcos**2+\
        -1.47613187e-06*1*x1*Psin**3+\
        -5.23520966e-10*1*x1*Psin**2*Pcos+\
        -1.86621602e-06*1*x1*Psin*Pcos**2+\
        +1.00443071e-09*1*x1*Pcos**3+\
        -6.89071790e-13*7*x0**6+\
        -1.23914182e-15*6*x0**5*x1+\
        +3.12291661e-15*6*x0**5*Psin+\
        +1.08128342e-11*6*x0**5*Pcos+\
        -2.06721537e-12*5*x0**4*x1**2+\
        -9.37110256e-12*5*x0**4*x1*Psin+\
        +1.97807743e-14*5*x0**4*x1*Pcos+\
        -3.13363989e-11*5*x0**4*Psin**2+\
        -1.36184934e-14*5*x0**4*Psin*Pcos+\
        -7.67764449e-11*5*x0**4*Pcos**2+\
        -3.71742547e-15*4*x0**3*x1**3+\
        -1.04120244e-14*4*x0**3*x1**2*Psin+\
        +2.30674000e-11*4*x0**3*x1**2*Pcos+\
        -4.62962000e-14*4*x0**3*x1*Psin**2+\
        +7.10094721e-11*4*x0**3*x1*Psin*Pcos+\
        -1.01589005e-13*4*x0**3*x1*Pcos**2+\
        +2.08518115e-13*4*x0**3*Psin**3+\
        +2.86546264e-10*4*x0**3*Psin**2*Pcos+\
        +1.79097965e-13*4*x0**3*Psin*Pcos**2+\
        +2.91303785e-10*4*x0**3*Pcos**3+\
        -2.06721537e-12*3*x0**2*x1**4+\
        -1.87422051e-11*3*x0**2*x1**3*Psin+\
        +3.95615485e-14*3*x0**2*x1**3*Pcos+\
        -8.82422239e-11*3*x0**2*x1**2*Psin**2+\
        +5.61116357e-14*3*x0**2*x1**2*Psin*Pcos+\
        -1.27983464e-10*3*x0**2*x1**2*Pcos**2+\
        -2.23968484e-10*3*x0**2*x1*Psin**3+\
        +5.94053532e-13*3*x0**2*x1*Psin**2*Pcos+\
        -2.07663122e-10*3*x0**2*x1*Psin*Pcos**2+\
        +4.76163532e-13*3*x0**2*x1*Pcos**3+\
        -3.71742547e-15*2*x0*x1**5+\
        -3.01927987e-14*2*x0*x1**4*Psin+\
        +1.36962975e-11*2*x0*x1**4*Pcos+\
        -1.34266711e-13*2*x0*x1**3*Psin**2+\
        +9.08800920e-11*2*x0*x1**3*Psin*Pcos+\
        -1.61503698e-13*2*x0*x1**3*Pcos**2+\
        -1.47597153e-13*2*x0*x1**2*Psin**3+\
        +3.02240758e-10*2*x0*x1**2*Psin**2*Pcos+\
        +2.95519494e-14*2*x0*x1**2*Psin*Pcos**2+\
        +3.79701968e-10*2*x0*x1**2*Pcos**3+\
        -6.89071790e-13*1*x1**6+\
        -9.37110256e-12*1*x1**5*Psin+\
        +1.97807743e-14*1*x1**5*Pcos+\
        -5.69058250e-11*1*x1**4*Psin**2+\
        +6.97301291e-14*1*x1**4*Psin*Pcos+\
        -5.12070188e-11*1*x1**4*Pcos**2+\
        -1.98148081e-10*1*x1**3*Psin**3+\
        +4.16904430e-13*1*x1**3*Psin**2*Pcos+\
        -2.85124332e-10*1*x1**3*Psin*Pcos**2+\
        +5.35213233e-13*1*x1**3*Pcos**3

    j11 = \
        +1.58511993e-08*1*x0**2+\
        +2.70048831e-05*2*x1*x0+\
        +1.51708814e-04*1*x0*Psin+\
        -7.05327065e-08*1*x0*Pcos+\
        +1.58511993e-08*3*x1**2+\
        +1.93314336e-08*2*x1*Psin+\
        -7.62998410e-05*2*x1*Pcos+\
        -1.77948983e-07*1*Psin**2+\
        +3.80139338e-04*1*Psin*Pcos+\
        +1.73485441e-07*1*Pcos**2+\
        -3.11635113e-12*1*x0**4+\
        -1.06013627e-08*2*x1*x0**3+\
        -4.30961361e-08*1*x0**3*Psin+\
        +1.09479997e-11*1*x0**3*Pcos+\
        -6.23270226e-12*3*x1**2*x0**2+\
        -1.38759571e-11*2*x1*x0**2*Psin+\
        +6.56829129e-08*2*x1*x0**2*Pcos+\
        +2.06159148e-10*1*x0**2*Psin**2+\
        +3.14318229e-07*1*x0**2*Psin*Pcos+\
        -3.35872182e-10*1*x0**2*Pcos**2+\
        -5.30068134e-09*4*x1**3*x0+\
        -4.30961361e-08*3*x1**2*x0*Psin+\
        +1.09479997e-11*3*x1**2*x0*Pcos+\
        -3.92658453e-07*2*x1*x0*Psin**2+\
        +3.83341741e-10*2*x1*x0*Psin*Pcos+\
        -3.73284881e-07*2*x1*x0*Pcos**2+\
        -1.47613187e-06*1*x0*Psin**3+\
        -5.23520966e-10*1*x0*Psin**2*Pcos+\
        -1.86621602e-06*1*x0*Psin*Pcos**2+\
        +1.00443071e-09*1*x0*Pcos**3+\
        -3.11635113e-12*5*x1**4+\
        -1.24119784e-11*4*x1**3*Psin+\
        +1.12933884e-08*4*x1**3*Pcos+\
        -1.02298543e-10*3*x1**2*Psin**2+\
        +9.18570279e-08*3*x1**2*Psin*Pcos+\
        -2.74144918e-11*3*x1**2*Pcos**2+\
        +4.95342895e-11*2*x1*Psin**3+\
        +9.19974098e-07*2*x1*Psin**2*Pcos+\
        -7.14441549e-10*2*x1*Psin*Pcos**2+\
        +7.24932022e-07*2*x1*Pcos**3+\
        -1.23914182e-15*1*x0**6+\
        -2.06721537e-12*2*x1*x0**5+\
        -9.37110256e-12*1*x0**5*Psin+\
        +1.97807743e-14*1*x0**5*Pcos+\
        -3.71742547e-15*3*x1**2*x0**4+\
        -1.04120244e-14*2*x1*x0**4*Psin+\
        +2.30674000e-11*2*x1*x0**4*Pcos+\
        -4.62962000e-14*1*x0**4*Psin**2+\
        +7.10094721e-11*1*x0**4*Psin*Pcos+\
        -1.01589005e-13*1*x0**4*Pcos**2+\
        -2.06721537e-12*4*x1**3*x0**3+\
        -1.87422051e-11*3*x1**2*x0**3*Psin+\
        +3.95615485e-14*3*x1**2*x0**3*Pcos+\
        -8.82422239e-11*2*x1*x0**3*Psin**2+\
        +5.61116357e-14*2*x1*x0**3*Psin*Pcos+\
        -1.27983464e-10*2*x1*x0**3*Pcos**2+\
        -2.23968484e-10*1*x0**3*Psin**3+\
        +5.94053532e-13*1*x0**3*Psin**2*Pcos+\
        -2.07663122e-10*1*x0**3*Psin*Pcos**2+\
        +4.76163532e-13*1*x0**3*Pcos**3+\
        -3.71742547e-15*5*x1**4*x0**2+\
        -3.01927987e-14*4*x1**3*x0**2*Psin+\
        +1.36962975e-11*4*x1**3*x0**2*Pcos+\
        -1.34266711e-13*3*x1**2*x0**2*Psin**2+\
        +9.08800920e-11*3*x1**2*x0**2*Psin*Pcos+\
        -1.61503698e-13*3*x1**2*x0**2*Pcos**2+\
        -1.47597153e-13*2*x1*x0**2*Psin**3+\
        +3.02240758e-10*2*x1*x0**2*Psin**2*Pcos+\
        +2.95519494e-14*2*x1*x0**2*Psin*Pcos**2+\
        +3.79701968e-10*2*x1*x0**2*Pcos**3+\
        -6.89071790e-13*6*x1**5*x0+\
        -9.37110256e-12*5*x1**4*x0*Psin+\
        +1.97807743e-14*5*x1**4*x0*Pcos+\
        -5.69058250e-11*4*x1**3*x0*Psin**2+\
        +6.97301291e-14*4*x1**3*x0*Psin*Pcos+\
        -5.12070188e-11*4*x1**3*x0*Pcos**2+\
        -1.98148081e-10*3*x1**2*x0*Psin**3+\
        +4.16904430e-13*3*x1**2*x0*Psin**2*Pcos+\
        -2.85124332e-10*3*x1**2*x0*Psin*Pcos**2+\
        +5.35213233e-13*3*x1**2*x0*Pcos**3+\
        -1.23914182e-15*7*x1**6+\
        -1.66578576e-14*6*x1**5*Psin+\
        +1.44173163e-12*6*x1**5*Pcos+\
        -8.79705113e-14*5*x1**4*Psin**2+\
        +1.98706199e-11*5*x1**4*Psin*Pcos+\
        -5.99146934e-14*5*x1**4*Pcos**2+\
        -2.97065567e-13*4*x1**3*Psin**3+\
        +9.31557039e-11*4*x1**3*Psin**2*Pcos+\
        -3.26695118e-13*4*x1**3*Psin*Pcos**2+\
        +6.25777798e-11*4*x1**3*Pcos**3

    jacobian0 = np.concatenate((j00, j01), axis=-1)
    jacobian1 = np.concatenate((j10, j11), axis=-1)
    return concatenate((jacobian0, jacobian1), axis=-2)
    
"""archbeam = ArchBeamSSM(P=1)

state = array([[0.0], [0.0]])  # some state [x, v]
tau = np.pi  # adimensional time or the phase

# Print outputs for various terms
#print("linear coefficient ", archbeam.linear_coefficient.shape, " = \n", archbeam.linear_coefficient)
#print("linear term ", archbeam.linear_term(state).shape, " = \n", archbeam.linear_term(state))
print("nonlinear term ", archbeam.nonlinear_term(state, tau).shape, " = \n", archbeam.nonlinear_term(state, tau))
#print("jacobian nonlinear_term ", archbeam.jacobian_nonlinear_term(state, tau).shape, " = \n", archbeam.jacobian_nonlinear_term(state, tau))
#print("external term ", archbeam.external_term(tau).shape, " = \n", archbeam.external_term(tau))
#print("all terms = \n", archbeam.all_terms(state, tau))

# Test with vectorized state
number_of_time_samples = 16
state_vectorized = np.hstack((np.arange(number_of_time_samples).reshape(number_of_time_samples, 1, 1)*0+0, 
                              np.zeros((number_of_time_samples,)).reshape(number_of_time_samples, 1, 1)*0)) # some sequence of states

tau_vectorized = np.linspace(0, 2*np.pi, number_of_time_samples, endpoint=False)

#print("sequence of states", state_vectorized.shape)
print("sequence of nonlinear terms", archbeam.nonlinear_term(state_vectorized, tau_vectorized).shape)
#print("sequence of external terms", archbeam.external_term(tau_vectorized).shape)
print("sequence of jacobians of the nonlinear terms", archbeam.jacobian_nonlinear_term(state_vectorized, tau_vectorized).shape, "\n", 
      np.round(archbeam.jacobian_nonlinear_term(state_vectorized, tau_vectorized),9))"""