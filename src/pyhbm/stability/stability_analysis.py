from dataclasses import dataclass

import numpy as np
from numpy import linalg
from numpy.typing import NDArray
from scipy.linalg import expm

from ..dynamical_system import (
    FirstOrderODE
)


@dataclass(frozen=True)
class StabilityReport:
    multipliers: NDArray[np.complexfloating]
    is_stable: bool


class FloquetAnalyzer:
    def __init__(self, time_domain_ode: FirstOrderODE = None) -> None:
        
        if time_domain_ode is None:
            raise ValueError("time_domain_ode must be provided to FloquetAnalyzer for computing stability")
        
        self.time_domain_ode = time_domain_ode
        
        print("FloquetAnalyzer: stability analysis is highly sensitive to the time resolution of the time series")

    def compute_monodromy_matrix(
        self,
        time_series,
        adimensional_time_samples,
        omega
    ) -> NDArray[np.complexfloating]:
        """
        Compute the monodromy matrix by propagating the variational equation
        over one period.
        
        The variational equation is: dPhi/dtau = J(tau) * Phi
        where J(tau) is the Jacobian of the ODE evaluated on the periodic orbit.
        
        The monodromy matrix M = Phi(T) is computed via time-ordered exponential.
        """
        
        if omega == 0:
            return np.eye(1) * 0
        
        n_states = time_series.shape[1]
        n_steps = len(adimensional_time_samples)
        
        dt = 2 * np.pi / n_steps / omega
        
        Phi = np.eye(n_states, dtype=complex)
        
        for i, tau in enumerate(adimensional_time_samples):
            state = time_series[i:i+1]
            J = self.time_domain_ode.compute_jacobian(state, tau)
            J = np.squeeze(J, axis=0)
            Phi = expm(J * dt) @ Phi
            
        return Phi
        
        # J = self.time_domain_ode.compute_jacobian(time_series[0:1], adimensional_time_samples[0])
        # return expm(J * dt) @ Phi

    def analyze(
        self,
        time_series,
        adimensional_time_samples,
        omega,
        tolerance: float = 1e-10
    ) -> StabilityReport:
        """
        Compute Floquet multipliers and determine stability.
        
        A periodic orbit is asymptotically stable if all Floquet multipliers
        (except the trivial one ~1) have magnitude less than 1.
        """
        M = self.compute_monodromy_matrix(time_series, adimensional_time_samples, omega)
        
        multipliers = linalg.eigvals(M)
        
        trivial_idx = np.argmin(np.abs(multipliers - 1))
        nontrivial_multipliers = np.delete(multipliers, trivial_idx)
        
        is_stable = bool(np.all(np.abs(nontrivial_multipliers) < 1 - tolerance))

        return StabilityReport(
            multipliers=multipliers,
            is_stable=is_stable,
        )

    @staticmethod
    def get_real_parts(report: StabilityReport) -> NDArray[np.floating]:
        return report.multipliers.real

    @staticmethod
    def get_imaginary_parts(report: StabilityReport) -> NDArray[np.floating]:
        return report.multipliers.imag

    @staticmethod
    def get_magnitudes(report: StabilityReport) -> NDArray[np.floating]:
        return np.abs(report.multipliers)

    @staticmethod
    def get_classification(report: StabilityReport, tolerance: float = 1e-9) -> str:
        
        magnitudes = np.abs(report.multipliers)
        
        max_mag = np.max(magnitudes)
        
        trivial_idx = np.argmin(np.abs(report.multipliers - 1))
        nontrivial_mags = np.delete(magnitudes, trivial_idx)
        
        if np.any(nontrivial_mags > 1 + tolerance):
            return "unstable"
        elif np.any(nontrivial_mags < 1 - tolerance):
            return "stable"
        else:
            return "neutral"
