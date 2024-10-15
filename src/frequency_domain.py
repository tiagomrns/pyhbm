import numpy as np
from numpy import array, concatenate, unique, hstack, array_split, vstack
from numpy.linalg import norm
from numpy.fft import rfft, irfft, fft


# %%
class Fourier(object):
    
    harmonics = unique(array([1,3])) # list of relevant harmonics
    polynomial_degree = 3
    
    number_of_harmonics = len(harmonics)
    harmonic_truncation_order = max(harmonics)
    number_of_time_samples = (polynomial_degree+1)*harmonic_truncation_order+1
    adimensional_time_samples = np.linspace(0, 2*np.pi, number_of_time_samples, endpoint=False)
    
    @staticmethod
    def update_class_variables(harmonics: np.ndarray, polynomial_degree: int):
        Fourier.harmonics = unique(harmonics) # list of relevant harmonics
        Fourier.polynomial_degree = polynomial_degree

        Fourier.number_of_harmonics = len(Fourier.harmonics)
        Fourier.harmonic_truncation_order = max(Fourier.harmonics)
        Fourier.number_of_time_samples = (Fourier.polynomial_degree+1)*Fourier.harmonic_truncation_order+1
        Fourier.adimensional_time_samples = np.linspace(0, 2*np.pi, Fourier.number_of_time_samples, endpoint=False)

    def __init__(self, coefficients: np.ndarray) -> None:
        """
        Creates a Fourier instance from a set of Fourier coefficients
        """
        assert coefficients.shape[0] == Fourier.number_of_harmonics, f"len(coefficients) = {len(coefficients)}"

        self.coefficients = coefficients
        # self.real_part = coefficients.real
        # self.imaginary_part = coefficients.imag
        self.time_series = None

    def new_from_time_series(time_series: np.ndarray):
        """
        Computes the Fourier coefficients (Fourier instance) of a time series by executing the real fast Fourier transform (rFFT)
        """
        all_coefficients = rfft(time_series, axis=0)
        new = Fourier(all_coefficients[Fourier.harmonics])
        new.time_series = time_series
        return new

    def compute_time_series(self) -> None:
        shape = list(self.coefficients.shape)
        shape[0] = Fourier.harmonic_truncation_order + 1
        new_coeff = np.zeros(shape, dtype = complex)
        new_coeff[Fourier.harmonics] = self.coefficients
        self.time_series = irfft(new_coeff, axis=0, n=Fourier.number_of_time_samples)

    def adimensional_time_derivative(self):
        return Fourier(coefficients = array([self.coefficients[n] * Fourier.harmonics[n] for n in Fourier.harmonics]) * 1j)

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
        R = np.vstack(self.coefficients.real)
        I = np.vstack(self.coefficients.imag)
        return np.vstack((R, I))
    
    def new_from_RI(RI: np.ndarray):
        complex_dimension = len(RI) // 2
        fourier_C = RI[:complex_dimension] + 1j*RI[complex_dimension:]
        return Fourier(array(array_split(fourier_C, Fourier.number_of_harmonics)))


#%%

class FourierOmegaPoint(object):
    def __init__(self, fourier: Fourier, omega: float):
        self.fourier = fourier
        self.omega = omega
        
    @staticmethod
    def new_from_RI_omega(RI_omega: np.ndarray):
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
    
    @staticmethod
    def to_RI_omega(x):
        if isinstance(x, np.ndarray):
            return x
        
        fourier = x.fourier.to_RI()
        return vstack((fourier, x.omega))
        
        


#%%

class JacobianFourier(object):
    
    polynomial_degree = Fourier.polynomial_degree - 1
    harmonics_state = Fourier.harmonics[:, np.newaxis] - Fourier.harmonics
    harmonics_state_conj = Fourier.harmonics[:, np.newaxis] + Fourier.harmonics
    harmonics = unique(concatenate((unique(harmonics_state), unique(harmonics_state_conj))))
    number_of_harmonics = len(harmonics)
    harmonic_truncation_order = max(harmonics)
    
    @staticmethod
    def update_class_variables():
        JacobianFourier.polynomial_degree = Fourier.polynomial_degree - 1
        JacobianFourier.harmonics_state = Fourier.harmonics[:, np.newaxis] - Fourier.harmonics
        JacobianFourier.harmonics_state_conj = Fourier.harmonics[:, np.newaxis] + Fourier.harmonics
        JacobianFourier.harmonics = unique(concatenate((unique(JacobianFourier.harmonics_state), unique(JacobianFourier.harmonics_state_conj))))
        JacobianFourier.number_of_harmonics = len(JacobianFourier.harmonics)
        JacobianFourier.harmonic_truncation_order = max(JacobianFourier.harmonics)

    def __init__(self, RR: np.ndarray, IR: np.ndarray, RI: np.ndarray, II: np.ndarray) -> None:
        self.RR = RR # Derivative of real part wrt real part
        self.IR = IR # Derivative of imag part wrt real part
        self.RI = RI # Derivative of real part wrt imag part
        self.II = II # Derivative of imag part wrt imag part

    def new_given_all_coefficients(all_coefficients: np.ndarray) -> None:

        state = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state]) # row by row
        state_conj = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state_conj])
        state_real = hstack(concatenate(state + state_conj, axis=1)) / Fourier.number_of_time_samples
        state_imag = hstack(concatenate(state - state_conj, axis=1)) / Fourier.number_of_time_samples

        return JacobianFourier(RR = state_real.real, IR = state_real.imag, RI = -state_imag.imag, II = state_imag.real)

    def new_from_time_series(time_series: np.ndarray):
        """
        Computes the Fourier coefficients (Fourier instance) of a time series by executing the real fast Fourier transform (rFFT)
        """
        all_coefficients = fft(time_series, axis=0)
        new = Fourier(all_coefficients[Fourier.harmonics])
        new.time_series = time_series
        return JacobianFourier.new_given_all_coefficients(all_coefficients)


#%% 
class FrequencyDomainFirstOrderODE(object):
    def __init__(self, first_order_ode) -> None:
        self.ode = first_order_ode
        self.I = np.eye(self.ode.dimension)
        self.complex_dimension = Fourier.number_of_harmonics * self.ode.dimension
        self.real_dimension = self.complex_dimension * 2
        self.linear_coefficients_real = np.kron(np.eye(Fourier.number_of_harmonics), self.ode.linear_coefficient)
        self.time_derivative_coefficients_imaginary = np.kron(np.diag(Fourier.harmonics), self.I)

        self.external_term = self.compute_external_force()
        self.linear = self.compute_linear_term()
        self.jacobian_linear_term = self.compute_jacobian_linear_term()

    # External Force
    def compute_external_force(self) -> Fourier:
        external_term_time_series = self.ode.external_term(Fourier.adimensional_time_samples)
        return Fourier.new_from_time_series(external_term_time_series)

    # Linear Term
    def compute_linear_term(self) -> Fourier:
        linear_coefficients = np.array([self.ode.linear_coefficient - self.I * n * 1j for n in Fourier.harmonics])
        return Fourier(linear_coefficients)

    # Linear Jacobian
    def compute_jacobian_linear_term(self) -> JacobianFourier:
        state = np.kron(np.eye(Fourier.number_of_harmonics), self.ode.linear_coefficient)
        state_conj = np.kron(np.where(JacobianFourier.harmonics_state_conj == 0, 1, 0), self.ode.linear_coefficient)

        RR = state + state_conj
        II = state - state_conj
        IR = np.kron(np.diag(Fourier.harmonics), -np.eye(self.ode.dimension))
        RI = -IR

        return JacobianFourier(RR, IR, RI, II)

    # Nonlinear Term
    def compute_nonlinear_term(self, state: Fourier) -> Fourier:
        state.compute_time_series()
        fnl_time_series = self.ode.nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return Fourier.new_from_time_series(fnl_time_series)

    # Residue
    def compute_residue(self, state: Fourier, omega: float) -> Fourier:
        nonlinear_term = self.compute_nonlinear_term(state)
        linear_term_coefficients = (self.linear.coefficients.real + omega * 1j * self.linear.coefficients.imag) @ state.coefficients
        residue_coefficients = linear_term_coefficients + nonlinear_term.coefficients + self.external_term.coefficients
        return Fourier(residue_coefficients)

    # Residue in Real-Imaginary Format
    def compute_residue_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        return self.compute_residue(x.fourier, x.omega).to_RI()

    # Jacobian of Nonlinear Term
    def compute_jacobian_nonlinear_term(self, state: Fourier) -> JacobianFourier:
        dfnldq_time_series = self.ode.jacobian_nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return JacobianFourier.new_from_time_series(dfnldq_time_series)

    # Jacobian of Residue in Real-Imaginary Format
    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> np.ndarray:

        jacobian_nonlinear_term = self.compute_jacobian_nonlinear_term(x.fourier)

        J_RR = jacobian_nonlinear_term.RR + self.jacobian_linear_term.RR
        J_RI = jacobian_nonlinear_term.RI + self.jacobian_linear_term.RI * x.omega
        J_IR = jacobian_nonlinear_term.IR + self.jacobian_linear_term.IR * x.omega
        J_II = jacobian_nonlinear_term.II + self.jacobian_linear_term.II

        return np.block([[J_RR, J_RI], [J_IR, J_II]])

    # Derivative of Residue with respect to omega
    def compute_derivative_wrt_omega(self, state: Fourier) -> np.ndarray:
        return self.linear.coefficients.imag @ state.coefficients * 1j
    
    # Derivative of Residue with respect to omega in Real-Imaginary Format
    def compute_derivative_wrt_omega_RI(self, state: Fourier) -> np.ndarray:
        derivative_wrt_omega = self.compute_derivative_wrt_omega(state)
        R = np.vstack(derivative_wrt_omega.real)
        I = np.vstack(derivative_wrt_omega.imag)
        return np.vstack((R, I))


# Test

"""import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples/duffing')))
from dynamical_system import *

duffing = Duffing(c=1, k=2, beta=1, P=2)

q1 = 2 + 1j
q3 = 0 + 0j
omega = 10

z1 = np.array([[1],[1j*omega]]) * q1
z3 = np.array([[1],[3j*omega]]) * q3

x = FourierOmegaPoint(Fourier(array([z1,z3]) * Fourier.number_of_time_samples), omega)

duffing_frequency_domain = FrequencyDomainFirstOrderODE(duffing)

residue = duffing_frequency_domain.compute_residue_RI(x)

actual_residue = residue / Fourier.number_of_time_samples

print("actual_residue =\n",  np.round(actual_residue[1::2], 9).ravel())

jacobian = duffing_frequency_domain.compute_jacobian_of_residue_RI(x) 

actual_jacobian = jacobian

print("actual_jacobian =\n", np.round(actual_jacobian[1::2,0::2], 9))"""