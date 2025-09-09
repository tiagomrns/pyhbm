import numpy as np
from numpy import array, concatenate, unique, hstack, array_split, vstack, einsum, pi, linspace, zeros, eye, kron, diag, where, block, zeros_like
from numpy.fft import rfft, irfft, fft


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
        Fourier.harmonics = unique(harmonics) # list of relevant harmonics
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
        
    def new_from_time_series(time_series: array):
        """
        Computes the Fourier coefficients (Fourier instance) of a time series by executing the fast Fourier transform (FFT)
        """
        all_coefficients = fft(time_series, axis=0)
        new = Fourier(all_coefficients[Fourier.harmonics])
        new.time_series = time_series
        return new

    def new_from_time_series_real(time_series: array):
        """
        Computes the Fourier coefficients (Fourier instance) of a time series by executing the real fast Fourier transform (rFFT)
        """
        all_coefficients = rfft(time_series, axis=0)
        new = Fourier(all_coefficients[Fourier.harmonics])
        new.time_series = time_series
        return new
    
    def compute_time_series(self) -> None:
        shape = list(self.coefficients.shape)
        shape[0] = 2 * Fourier.harmonic_truncation_order + 1
        new_coeff = zeros(shape, dtype = complex)
        new_coeff[Fourier.harmonics] = self.coefficients
        self.time_series = irfft(new_coeff, axis=0, n=Fourier.number_of_time_samples)

    def compute_time_series_real(self) -> None:
        shape = list(self.coefficients.shape)
        shape[0] = Fourier.harmonic_truncation_order + 1
        new_coeff = zeros(shape, dtype = complex)
        new_coeff[Fourier.harmonics] = self.coefficients
        self.time_series = irfft(new_coeff, axis=0, n=Fourier.number_of_time_samples)

    def adimensional_time_derivative(self):
        return Fourier(coefficients = einsum('i,ijk->ijk', Fourier.harmonics, self.coefficients) * 1j)

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
        assert 1 in Fourier.harmonics, "Harmonic 1 is not in the provided list of harmonics"
        z1 = first_harmonic * 0.5 * Fourier.number_of_time_samples
        zz = zeros_like(z1)
        zz_fill = [zz if h !=1 else z1 for h in Fourier.harmonics]
        return Fourier(array(zz_fill))
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
        adimensional_time_derivative = self.fourier.adimensional_time_derivative()
        return vstack((adimensional_time_derivative.to_RI(), 0.0))
    
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

    def new_given_all_coefficients(all_coefficients: array) -> None:

        state = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state]) # row by row
        state_conj = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state_conj])
        state_real = hstack(concatenate(state + state_conj, axis=1)) / Fourier.number_of_time_samples
        state_imag = hstack(concatenate(state - state_conj, axis=1)) / Fourier.number_of_time_samples

        return JacobianFourier(RR = state_real.real, RI = -state_imag.imag, IR = state_real.imag, II = state_imag.real)

    def new_from_time_series(time_series: array):
        """
        Computes the JacobianFourier coefficients given a time series by executing the fast Fourier transform (FFT)
        """
        all_coefficients = fft(time_series, axis=0)
        #new = Fourier(all_coefficients[Fourier.harmonics])
        #new.time_series = time_series
        return JacobianFourier.new_given_all_coefficients(all_coefficients)


#%% 
class FrequencyDomainFirstOrderODE(object):
    def __init__(self, first_order_ode) -> None:
        self.ode = first_order_ode
        self.I = eye(self.ode.dimension)
        self.complex_dimension = Fourier.number_of_harmonics * self.ode.dimension
        self.real_dimension = self.complex_dimension * 2
        self.time_derivative_coefficients_imaginary = kron(diag(Fourier.harmonics), self.I)

        self.external_term = self.compute_external_force()
        self.linear = self.compute_linear_term()
        self.jacobian_linear_term = self.compute_jacobian_linear_term()

    # Linear Term
    def compute_linear_term(self) -> Fourier:
        linear_coefficients = array([self.ode.linear_coefficient - self.I * n * 1j for n in Fourier.harmonics])
        return Fourier(linear_coefficients)

    # Linear Jacobian
    def compute_jacobian_linear_term(self) -> JacobianFourier:
        state = kron(eye(Fourier.number_of_harmonics), self.ode.linear_coefficient)
        state_conj = kron(where(JacobianFourier.harmonics_state_conj == 0, 1, 0), self.ode.linear_coefficient)

        RR = state + state_conj
        II = state - state_conj
        RI = kron(diag(Fourier.harmonics), eye(self.ode.dimension))
        IR = -RI 

        return JacobianFourier(RR=RR, RI=RI, IR=IR, II=II)
    
    """
    # The following methods are for systems where the external force and nonlinear terms are complex-valued
    # and the Fourier coefficients are computed using the FFT (FFT).
    """
    
    # External Force
    def compute_external_force(self) -> Fourier:
        external_term_time_series = self.ode.external_term(Fourier.adimensional_time_samples)
        return Fourier.new_from_time_series(external_term_time_series)

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
    def compute_residue_RI(self, x: FourierOmegaPoint) -> array:
        return self.compute_residue(x.fourier, x.omega).to_RI()
    
    """
    # The following methods are for systems where the external force and nonlinear terms are real-valued
    # and the Fourier coefficients are computed using the real FFT (rFFT).
    """
    
    # External Force Real
    def compute_external_force_real(self) -> Fourier:
        external_term_time_series = self.ode.external_term(Fourier.adimensional_time_samples)
        return Fourier.new_from_time_series_real(external_term_time_series)

    # Nonlinear Term Real
    def compute_nonlinear_term_real(self, state: Fourier) -> Fourier:
        state.compute_time_series_real()
        fnl_time_series = self.ode.nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return Fourier.new_from_time_series_real(fnl_time_series)

    # Residue Real
    def compute_residue_real(self, state: Fourier, omega: float) -> Fourier:
        nonlinear_term = self.compute_nonlinear_term_real(state)
        linear_term_coefficients = (self.linear.coefficients.real + omega * 1j * self.linear.coefficients.imag) @ state.coefficients
        residue_coefficients = linear_term_coefficients + nonlinear_term.coefficients + self.external_term.coefficients
        return Fourier(residue_coefficients)

    # Residue Real in Real-Imaginary Format
    def compute_residue_real_RI(self, x: FourierOmegaPoint) -> array:
        return self.compute_residue_real(x.fourier, x.omega).to_RI()
    
    """
    # The following methods compute the Jacobian of the nonlinear term and the residue.
    # These methods are valid for both complex-valued and real-valued systems.
    """

    # Jacobian of Nonlinear Term
    def compute_jacobian_nonlinear_term(self, state: Fourier) -> JacobianFourier:
        dfnldq_time_series = self.ode.jacobian_nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return JacobianFourier.new_from_time_series(dfnldq_time_series)

    # Jacobian of Residue in Real-Imaginary Format
    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:

        jacobian_nonlinear_term = self.compute_jacobian_nonlinear_term(x.fourier)

        J_RR = jacobian_nonlinear_term.RR + self.jacobian_linear_term.RR
        J_RI = jacobian_nonlinear_term.RI + self.jacobian_linear_term.RI * x.omega
        J_IR = jacobian_nonlinear_term.IR + self.jacobian_linear_term.IR * x.omega
        J_II = jacobian_nonlinear_term.II + self.jacobian_linear_term.II

        return block([[J_RR, J_RI], [J_IR, J_II]])

    # Derivative of Residue with respect to omega
    def compute_derivative_wrt_omega(self, state: Fourier) -> array:
        return self.linear.coefficients.imag @ state.coefficients * 1j
    
    # Derivative of Residue with respect to omega in Real-Imaginary Format
    def compute_derivative_wrt_omega_RI(self, state: Fourier) -> array:
        derivative_wrt_omega = self.compute_derivative_wrt_omega(state)
        R = vstack(derivative_wrt_omega.real)
        I = vstack(derivative_wrt_omega.imag)
        return vstack((R, I))

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