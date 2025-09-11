import numpy as np
from numpy import array, concatenate, unique, hstack, array_split, vstack, einsum, pi, linspace, zeros, eye, kron, diag, where, block, zeros_like
from scipy.linalg import block_diag
from numpy.fft import rfft, irfft, fft, ifft


# %%
class Fourier(object):
    
    harmonics = unique(array([1,3])) # list of relevant harmonics
    polynomial_degree = 3
    
    number_of_harmonics = len(harmonics)
    harmonic_truncation_order = max(abs(harmonics))
    number_of_time_samples = (polynomial_degree+1)*harmonic_truncation_order+1
    adimensional_time_samples = linspace(0, 2*pi, number_of_time_samples, endpoint=False)
    
    @staticmethod
    def update_class_variables(harmonics: array, polynomial_degree: int):
        indexes = sorted(np.unique(harmonics, return_index=True)[1])
        Fourier.harmonics = array(harmonics)[indexes] # list of relevant harmonics
        Fourier.polynomial_degree = polynomial_degree

        Fourier.number_of_harmonics = len(Fourier.harmonics)
        Fourier.harmonic_truncation_order = max(abs(Fourier.harmonics))
        Fourier.number_of_time_samples = (Fourier.polynomial_degree+1)*Fourier.harmonic_truncation_order+1
        Fourier.adimensional_time_samples = linspace(0, 2*pi, Fourier.number_of_time_samples, endpoint=False)
        
    def __init__(self, coefficients: array) -> None:
        """
        Creates a Fourier instance from a set of Fourier coefficients
        """
        assert coefficients.shape[0] == Fourier.number_of_harmonics, \
            f"Number of harmonics is {Fourier.number_of_harmonics}, but {len(coefficients)} coefficients were provided."

        self.coefficients = coefficients
        # self.real_part = coefficients.real
        # self.imaginary_part = coefficients.imag
        self.time_series = None
        self.adimensional_time_derivative = None

    def compute_adimensional_time_derivative(self):
        self.adimensional_time_derivative = einsum('i,ijk->ijk', Fourier.harmonics, self.coefficients) * 1j
    
    def get_adimensional_time_derivative(self):
        if self.adimensional_time_derivative is None:
            self.compute_adimensional_time_derivative()
        return self.adimensional_time_derivative
    
    def new_from_time_series(time_series: array):
        pass
    
    def compute_time_series(self):
        pass

    def __add__(self, other):
        return Fourier(coefficients = self.coefficients + other.coefficients)
    
    def __sub__(self, other):
        return Fourier(coefficients = self.coefficients - other.coefficients)

    def matmul(self, other):
        return Fourier(coefficients = self.coefficients @ other.coefficients)

    def __mul__(self, other: float):
        return Fourier(coefficients = self.coefficients * other)

    def __rmul__(self, other: float):
        return Fourier(coefficients = self.coefficients * other)
    
    def to_RI(self):
        R = vstack(self.coefficients.real)
        I = vstack(self.coefficients.imag)
        return vstack((R, I))
    
    def new_from_RI(RI: array):
        complex_dimension = len(RI) // 2
        fourier_C = RI[:complex_dimension] + 1j*RI[complex_dimension:]
        return Fourier(array(array_split(fourier_C, Fourier.number_of_harmonics)))
    
    @staticmethod
    def zeros(dimension: int):
        return Fourier(zeros((Fourier.number_of_harmonics, dimension, 1), dtype=complex))
    
    @staticmethod
    def new_from_first_harmonic(first_harmonic: array):
        assert 1 in Fourier.harmonics, "Fourier: Harmonic 1 is not in the list of harmonics"
        z1 = first_harmonic * 0.5 * Fourier.number_of_time_samples
        zz = zeros_like(z1)
        zz_fill = [zz if h !=1 else z1 for h in Fourier.harmonics]
        return Fourier(array(zz_fill))
    
class Fourier_Real(Fourier):
    def new_from_time_series(time_series: array):
        """
        Computes the Fourier coefficients (Fourier instance) of a time series by executing the Real Fast Fourier transform (rFFT)
        """
        all_coefficients = rfft(time_series, axis=0)
        new = Fourier(all_coefficients[Fourier.harmonics])
        new.time_series = time_series
        return new
    
    def compute_time_series(self) -> None:
        shape = list(self.coefficients.shape)
        shape[0] = Fourier.harmonic_truncation_order + 1
        new_coeff = zeros(shape, dtype = complex)
        new_coeff[Fourier.harmonics] = self.coefficients
        # inverse of Real FFT
        self.time_series = irfft(new_coeff, axis=0, n=Fourier.number_of_time_samples)
        
class Fourier_Complex(Fourier):
    def new_from_time_series(time_series: array):
        """
        Computes the Fourier coefficients (Fourier instance) of a time series by executing the Fast Fourier transform (FFT)
        """
        all_coefficients = fft(time_series, axis=0)
        new = Fourier(all_coefficients[Fourier.harmonics])
        new.time_series = time_series
        return new
    
    def compute_time_series(self) -> None:
        shape = list(self.coefficients.shape)
        shape[0] = 2 * Fourier.harmonic_truncation_order + 1
        new_coeff = zeros(shape, dtype = complex)
        new_coeff[Fourier.harmonics] = self.coefficients
        # inverse of FFT
        self.time_series = ifft(new_coeff, axis=0, n=Fourier.number_of_time_samples)
    
#%%

class FourierOmegaPoint(object):
    def __init__(self, fourier: Fourier, omega: float):
        self.fourier: Fourier = fourier
        self.omega: float = omega
        self.RI = None
        
    @staticmethod
    def new_from_RI_omega(RI_omega: array):
        # in case omega is not included in the array
        if len(RI_omega) % 2 == 0:
            RI_omega = vstack((RI_omega, 0))
            
        omega = RI_omega[-1,0]
        fourier = Fourier.new_from_RI(RI_omega[:-1])
        return FourierOmegaPoint(fourier=fourier, omega=omega)

    def __add__ (self, other):
        if isinstance(other, np.ndarray):
            other = FourierOmegaPoint.new_from_RI_omega(other)
        
        return FourierOmegaPoint(self.fourier + other.fourier, self.omega + other.omega)

    def __sub__ (self, other):
        if isinstance(other, np.ndarray):
            other = FourierOmegaPoint.new_from_RI_omega(other)
        
        return FourierOmegaPoint(self.fourier - other.fourier, self.omega - other.omega)
    
    def __mul__ (self, other: float):
        return FourierOmegaPoint(self.fourier*other, self.omega*other)
    
    def to_RI_omega(self):
        if self.RI is None:
            self.RI = vstack((self.fourier.to_RI(), self.omega))
            
        return self.RI
    
    @staticmethod
    def to_RI_omega_static(x):
        if isinstance(x, np.ndarray):
            return x
        
        return x.to_RI_omega()
    
    def adimensional_time_derivative_RI(self) -> array:
        adimensional_time_derivative = self.fourier.get_adimensional_time_derivative()
        R = vstack(adimensional_time_derivative.real)
        I = vstack(adimensional_time_derivative.imag)
        return vstack((R, I, 0.0))
    
    def zero_amplitude(dimension: int, omega: float):
        return FourierOmegaPoint(Fourier.zeros(dimension), omega)
    
    def new_from_first_harmonic(first_harmonic: array, omega: float):
        return FourierOmegaPoint(Fourier.new_from_first_harmonic(first_harmonic), omega)

#%%

class JacobianFourier(object):
    
    polynomial_degree = Fourier.polynomial_degree - 1
    harmonics_state = Fourier.harmonics[:, None] - Fourier.harmonics
    harmonics_state_conj = Fourier.harmonics[:, None] + Fourier.harmonics
    harmonics = unique(concatenate((unique(harmonics_state), unique(harmonics_state_conj))))
    number_of_harmonics = len(harmonics)
    harmonic_truncation_order = max(harmonics)
    
    @staticmethod
    def update_class_variables():
        JacobianFourier.polynomial_degree = Fourier.polynomial_degree - 1
        JacobianFourier.harmonics_state = Fourier.harmonics[:, None] - Fourier.harmonics
        JacobianFourier.harmonics_state_conj = Fourier.harmonics[:, None] + Fourier.harmonics
        JacobianFourier.harmonics = unique(concatenate((unique(JacobianFourier.harmonics_state), unique(JacobianFourier.harmonics_state_conj))))
        JacobianFourier.number_of_harmonics = len(JacobianFourier.harmonics)
        JacobianFourier.harmonic_truncation_order = max(JacobianFourier.harmonics)

    def __init__(self, RR: array, RI: array, IR: array, II: array) -> None:
        self.RR = RR # Derivative of real part wrt real part
        self.RI = RI # Derivative of real part wrt imag part
        self.IR = IR # Derivative of imag part wrt real part
        self.II = II # Derivative of imag part wrt imag part
        
    def new_from_time_series(time_series: array):
        pass
    
    def new_given_all_coefficients(all_coefficients: array):
        pass

class JacobianFourier_Real(JacobianFourier):
    
    def new_from_time_series(time_series: array):
        """
        Computes the JacobianFourier coefficients given a time series by executing the fast Fourier transform (FFT)
        """
        all_coefficients = fft(time_series, axis=0)
        return JacobianFourier_Real.new_given_all_coefficients(all_coefficients)
    
    def new_given_all_coefficients(all_coefficients: array):

        state = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state]) # row by row
        state_conj = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state_conj])
        state_real = hstack(concatenate(state + state_conj, axis=1)) / Fourier.number_of_time_samples
        state_imag = hstack(concatenate(state - state_conj, axis=1)) / Fourier.number_of_time_samples
        
        return JacobianFourier_Real(RR = state_real.real, RI = -state_imag.imag, IR = state_real.imag, II = state_imag.real)

class JacobianFourier_Complex(JacobianFourier):
    
    def new_from_time_series(time_series: array):
        """
        Computes the JacobianFourier coefficients given a time series by executing the fast Fourier transform (FFT)
        """
        all_coefficients = fft(time_series, axis=0)
        return JacobianFourier_Complex.new_given_all_coefficients(all_coefficients)
    
    def new_given_all_coefficients(all_coefficients: array):
        state_blocks = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state])
        state = hstack(concatenate(state_blocks, axis=1)) / Fourier.number_of_time_samples
        return JacobianFourier_Complex(RR = state.real, RI = -state.imag, IR = state.imag, II = state.real)

#%% 

class FirstOrderODE:
  """
  Base class for dynamical systems.
  Class that implements the dynamics
  zdot = omega z' = f(z, tau) 
  tau = omega * t
  
  """
  
  is_real_valued: bool = True
  
  def __init__(self):
    self.linear_coefficient: int = 1
    self.dimension: int = 1
    self.polynomial_degree: int = 1
    
  def external_term(self, adimensional_time: array) -> array:
    raise NotImplementedError("This method should be overridden by subclasses.")

  def linear_term(self, state: array) -> array:
    raise NotImplementedError("This method should be overridden by subclasses.")

  def nonlinear_term(self, state: array, adimensional_time: array) -> array:
    raise NotImplementedError("This method should be overridden by subclasses.")

class FrequencyDomainFirstOrderODE(object):
    def __init__(self, first_order_ode: FirstOrderODE) -> None:
        self.ode = first_order_ode
        self.complex_dimension = Fourier.number_of_harmonics * self.ode.dimension
        self.real_dimension = self.complex_dimension * 2

        self.external_term = self.compute_external_force()
        self.jacobian_linear_term = self.compute_jacobian_linear_term()

    # Residue in Real-Imaginary Format
    def compute_residue_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        nonlinear_term = self.compute_nonlinear_term(state)
        linear_term_coefficients = self.ode.linear_coefficient @ state.coefficients - state.get_adimensional_time_derivative() * x.omega
        residue_coefficients = linear_term_coefficients + nonlinear_term.coefficients + self.external_term.coefficients
        return Fourier(residue_coefficients).to_RI()
    
    # Derivative of Residue with respect to omega in Real-Imaginary Format
    def compute_derivative_wrt_omega_RI(self, state: Fourier) -> array:
        derivative_wrt_omega = -state.get_adimensional_time_derivative()
        R = vstack(derivative_wrt_omega.real)
        I = vstack(derivative_wrt_omega.imag)
        return vstack((R, I))
    
    """
    # The following methods must be specified separately for real-valued and complex-valued systems.
    """
    
    def compute_jacobian_linear_term(self) -> JacobianFourier:
        pass
    
    def compute_external_force(self) -> Fourier:
        pass

    def compute_nonlinear_term(self, state: Fourier) -> Fourier:
        pass
    
    def compute_jacobian_nonlinear_term(self, state: Fourier) -> JacobianFourier:
        pass

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:
        pass

class FrequencyDomainFirstOrderODE_Real(FrequencyDomainFirstOrderODE):
    
    """
    # The following methods are for systems where the external force and nonlinear terms are real-valued
    # and the Fourier coefficients are computed using the Real FFT (rFFT).
    """
    
    # Linear Jacobian for Real-Valued Systems
    def compute_jacobian_linear_term(self) -> JacobianFourier_Real:

        state = kron(eye(Fourier.number_of_harmonics), self.ode.linear_coefficient)
        state_conj = kron(where(JacobianFourier.harmonics_state_conj == 0, 1, 0), self.ode.linear_coefficient)

        RR = state + state_conj
        II = state - state_conj
        self.jacobian_derivative_term = kron(diag(Fourier.harmonics), eye(self.ode.dimension))
        # RI = 0
        # IR = -RI = 0
        return JacobianFourier_Real(RR=RR, RI=None, IR=None, II=II)
    
    # External Force for Real-Valued Systems
    def compute_external_force(self) -> Fourier_Real:
        external_term_time_series = self.ode.external_term(Fourier.adimensional_time_samples)
        return Fourier_Real.new_from_time_series(external_term_time_series)

    # Nonlinear Term for Real-Valued Systems
    def compute_nonlinear_term(self, state: Fourier_Real) -> Fourier_Real:
        Fourier_Real.compute_time_series(state)
        fnl_time_series = self.ode.nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return Fourier_Real.new_from_time_series(fnl_time_series)
    
    def compute_jacobian_nonlinear_term(self, state: Fourier_Real) -> JacobianFourier_Real:
        dfnldq_time_series = self.ode.jacobian_nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return JacobianFourier_Real.new_from_time_series(dfnldq_time_series)

    # Jacobian of Residue for Real-Valued Systems in Real-Imaginary Format
    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:

        jacobian_nonlinear_term = self.compute_jacobian_nonlinear_term(x.fourier)
        aux = self.jacobian_derivative_term * x.omega

        J_RR = jacobian_nonlinear_term.RR + self.jacobian_linear_term.RR
        J_RI = jacobian_nonlinear_term.RI + aux
        J_IR = jacobian_nonlinear_term.IR - aux
        J_II = jacobian_nonlinear_term.II + self.jacobian_linear_term.II

        return block([[J_RR, J_RI], [J_IR, J_II]])

class FrequencyDomainFirstOrderODE_Complex(FrequencyDomainFirstOrderODE):
    
    """
    # The following methods are for systems where the external force and nonlinear terms are complex-valued
    # and the Fourier coefficients are computed using the FFT (FFT).
    """
    
    # Linear Jacobian for Complex-Valued Systems
    def compute_jacobian_linear_term(self) -> JacobianFourier_Complex:
        linear = kron(eye(Fourier.number_of_harmonics), self.ode.linear_coefficient)
        self.jacobian_derivative_term = kron(diag(Fourier.harmonics), eye(self.ode.dimension))
        # IR = -RI = linear.imag
        # II = RR = linear.real
        return JacobianFourier_Complex(RR = linear.real, RI = -linear.imag, IR = None, II = None)
    
    # External Force for Complex-Valued Systems
    def compute_external_force(self) -> Fourier_Complex:
        external_term_time_series = self.ode.external_term(Fourier.adimensional_time_samples)
        return Fourier_Complex.new_from_time_series(external_term_time_series)

    # Nonlinear Term for Complex-Valued Systems
    def compute_nonlinear_term(self, state: Fourier_Complex) -> Fourier_Complex:
        Fourier_Complex.compute_time_series(state)
        fnl_time_series = self.ode.nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return Fourier_Complex.new_from_time_series(fnl_time_series)
    
    # Jacobian of Nonlinear Term for Complex-Valued Systems
    def compute_jacobian_nonlinear_term(self, state: Fourier_Complex) -> JacobianFourier_Complex:
        dfnldq_time_series = self.ode.jacobian_nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return JacobianFourier_Complex.new_from_time_series(dfnldq_time_series)
    
    # Jacobian of Residue for Complex-Valued Systems in Real-Imaginary Format
    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:

        jacobian_nonlinear_term = self.compute_jacobian_nonlinear_term(x.fourier)
        aux = self.jacobian_linear_term.RI + self.jacobian_derivative_term * x.omega

        J_RR = jacobian_nonlinear_term.RR + self.jacobian_linear_term.RR
        J_RI = jacobian_nonlinear_term.RI + aux
        J_IR = jacobian_nonlinear_term.IR - aux
        J_II = jacobian_nonlinear_term.II + self.jacobian_linear_term.RR
        
        return block([[J_RR, J_RI], [J_IR, J_II]])

# Test

if __name__ == "__main__":

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples/duffing_forced_nonautonomous')))
    from dynamical_system import *

    duffing = DuffingForced(c=1, k=2, beta=1, P=2)

    q1 = 2 + 1j
    q3 = 0 + 0j
    omega = 10

    z1 = array([[1],[1j*omega]]) * q1
    z3 = array([[1],[3j*omega]]) * q3

    x = FourierOmegaPoint(Fourier(array([z1,z3]) * Fourier.number_of_time_samples), omega)

    duffing_frequency_domain = FrequencyDomainFirstOrderODE(duffing)

    residue = duffing_frequency_domain.compute_residue_RI(x)

    actual_residue = residue / Fourier.number_of_time_samples

    print("actual_residue =\n",  np.round(actual_residue[1::2], 9).ravel())

    jacobian = duffing_frequency_domain.compute_jacobian_of_residue_RI(x) 

    actual_jacobian = jacobian

    print("actual_jacobian =\n", np.round(actual_jacobian[1::2,0::2], 9))
# %%
