from dataclasses import dataclass
from typing import Optional
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from .stability_analysis import FloquetAnalyzer, StabilityReport

@dataclass
class SpecialPoint:
    """Represents a detected bifurcation point."""

    type: str
    index: int
    omega_crossing: Optional[float] = None
    multipliers: Optional[NDArray[np.complexfloating]] = None
    hopf_frequency: Optional[float] = None
    relative_period: Optional[float] = None
    rational_approx_relative_period: Optional[Fraction] = None

    def __str__(self, verbose=False) -> str:
        if not verbose:
            base = f"{self.type} at omega={self.omega_crossing:.4f}, index={self.index}"
            if self.type == 'Hopf':
                if self.relative_period is not None:
                    base += f", period_ratio={self.relative_period:.2e}"
                if self.rational_approx_relative_period is not None:
                    base += f", rational_approx={self.rational_approx_relative_period}"
            return base

        """Detailed string with all bifurcation data."""
        lines = [f"Bifurcation: {self.type}"]
        lines.append(f"  Index: {self.index}")
        lines.append(f"  omega: {self.omega_crossing:.6f}")
        
        if self.type == 'Hopf':
            if self.relative_period is not None:
                lines.append(f"  Relative Period (>=1): {self.relative_period:.6e}")
            if self.rational_approx_relative_period is not None:
                lines.append(f"  Rational Approximation: {self.rational_approx_relative_period}")
            if self.hopf_frequency is not None:
                lines.append(f"  Hopf frequency: {self.hopf_frequency:.6f}")
            if self.multipliers is not None:
                lines.append(f"  Multipliers:")
                for i, m in enumerate(self.multipliers):
                    lines.append(f"    [{i}] {m.real:.6f} + {m.imag:.6f}j")
        
        return "\n".join(lines)


class BifurcationDetector:
    """
    Detects bifurcations in continuation solutions.

    Parameters
    ----------
    tolerance : float, optional
        Tolerance for detecting zero crossings. Default is 1e-6.
    commensurability_tolerance : float, optional
        Relative tolerance for commensurability detection. Default is 0.01 (1%).
    """

    def __init__(self, tolerance: float = 1e-6, commensurability_tolerance: float = 0.01):
        self.tolerance = tolerance
        self.commensurability_tolerance = commensurability_tolerance

    def _compute_unwrapped_angles(
        self,
        stability_reports: list['StabilityReport'],
        multiplier_index: int
    ) -> NDArray[np.floating]:
        """
        Compute cumulative unwrapped angles for a specific multiplier across all solutions.

        Parameters
        ----------
        stability_reports : list[StabilityReport]
            Pre-computed stability reports for each solution.
        multiplier_index : int
            Index of the multiplier to track.

        Returns
        -------
        NDArray[np.floating]
            Cumulative unwrapped angles (in radians) for each solution point.
        """
        raw_angles = np.array([np.angle(report.multipliers[multiplier_index]) 
                              for report in stability_reports])
        
        unwrapped = np.zeros_like(raw_angles)
        unwrapped[0] = raw_angles[0]
        
        for i in range(1, len(raw_angles)):
            delta = raw_angles[i] - raw_angles[i - 1]
            if delta > np.pi:
                delta -= 2 * np.pi
            elif delta < -np.pi:
                delta += 2 * np.pi
            unwrapped[i] = unwrapped[i - 1] + delta
        
        return unwrapped

    def detect_saddle_nodes(
        self,
        solution_set
    ) -> list['SpecialPoint']:
        """
        Detect saddle-node bifurcations.

        A saddle-node occurs when two solutions collide and annihilate.
        Detected by monitoring domega/dn for sign changes.

        Parameters
        ----------
        solution_set
            Continuation solutions.

        Returns
        -------
        list[SpecialPoint]
            List of detected saddle-node points.
        """
        saddle_nodes = []

        if len(solution_set.omega) < 3:
            return saddle_nodes

        omega_array = np.array(solution_set.omega)
        domega_dn = np.diff(omega_array)

        for i in range(len(domega_dn) - 1):
            product = domega_dn[i] * domega_dn[i + 1]

            if product < 0:
                saddle_node = SpecialPoint(
                    type='Saddle-Node',
                    index= i + 1,
                    omega_crossing=omega_array[i + 1],
                    multipliers=None,
                    hopf_frequency=None,
                    relative_period=None,
                    rational_approx_relative_period=None,
                )
                saddle_nodes.append(saddle_node)

        return saddle_nodes

    def detect_hopf(
        self,
        solution_set,
        stability_reports: list['StabilityReport'],
        unwrapped_angles: Optional[dict[int, NDArray[np.floating]]] = None
    ) -> list['SpecialPoint']:
        """
        Detect Hopf bifurcations.

        A Hopf bifurcation occurs when a pair of complex conjugate multipliers
        cross the unit circle (|mu| = 1).

        Parameters
        ----------
        solution_set
            Continuation solutions.
        stability_reports : list[StabilityReport]
            Pre-computed stability reports for each solution.

        Returns
        -------
        list[SpecialPoint]
            List of detected Hopf bifurcation points.
        """
        hopf_points = []

        if len(stability_reports) < 2:
            return hopf_points

        for i in range(len(stability_reports) - 1):
            mags_current = np.abs(stability_reports[i].multipliers) - 1
            mags_next = np.abs(stability_reports[i + 1].multipliers) - 1

            for j in range(len(mags_current)):
                mag_current = mags_current[j]
                mag_next = mags_next[j]

                if mag_current * mag_next < 0: # one on either side of the unit circle

                    # current point
                    mult_current = stability_reports[i].multipliers
                    omega_current = solution_set.omega[i]
                    
                    # next point
                    mult_next = stability_reports[i + 1].multipliers
                    omega_next = solution_set.omega[i + 1]
                    
                    # interpolate
                    delta = mag_current - mag_next # != 0, as mag_current * mag_next < 0
                    t_crossing = mag_current / delta
                    mult_crossing = mult_current + t_crossing * (mult_next - mult_current)
                    omega_crossing = omega_current +  t_crossing * (omega_next - omega_current)

                    if unwrapped_angles is not None and j in unwrapped_angles:
                        cumulative_angles = unwrapped_angles[j]
                        angle_hopf = cumulative_angles[i] + t_crossing * (cumulative_angles[i + 1] - cumulative_angles[i])
                    else:
                        angle_hopf = np.angle(mult_crossing[j])
                    
                    relative_frequency = angle_hopf / (2 * np.pi)

                    relative_period = 1.0 / relative_frequency if relative_frequency != 0.0 else None

                    hopf = SpecialPoint(
                        type='Hopf',
                        index= i + 1,
                        omega_crossing=omega_crossing,
                        multipliers=mult_crossing,
                        hopf_frequency = omega_crossing * relative_frequency,
                        relative_period = relative_period,
                        rational_approx_relative_period=rational_approx(relative_period)
                    )
                    hopf_points.append(hopf)

        return hopf_points

    def detect_all(
        self,
        solution_set,
        stability_reports: Optional[list[StabilityReport]] = None,
        freq_domain_ode=None
    ) -> list['SpecialPoint']:
        """
        Detect all bifurcation types.

        Parameters
        ----------
        solution_set
            Continuation solutions.
        stability_reports : list[StabilityReport], optional
            Pre-computed stability reports. If None, computes on-demand.
        freq_domain_ode : FrequencyDomainFirstOrderODE, optional
            Required if stability_reports is None.

        Returns
        -------
        list[SpecialPoint]
            List of all detected bifurcation points.
        """
        if stability_reports is None:
            if freq_domain_ode is None:
                raise ValueError("Either stability_reports or freq_domain_ode must be provided")
            analyzer = FloquetAnalyzer(freq_domain_ode.ode)
            stability_reports = []
            from ..frequency_domain import Fourier
            for idx, fourier in enumerate(solution_set.fourier):
                if fourier.time_series is None:
                    fourier.compute_time_series()
                report = analyzer.analyze(fourier.time_series, Fourier.adimensional_time_samples, solution_set.omega[idx])
                stability_reports.append(report)

        num_multipliers = len(stability_reports[0].multipliers)
        unwrapped_angles = {}
        for mult_idx in range(0, num_multipliers, 2):
            unwrapped_angles[mult_idx] = self._compute_unwrapped_angles(
                stability_reports, mult_idx
            )

        saddle_nodes = self.detect_saddle_nodes(solution_set)
        hopf_points = self.detect_hopf(solution_set, stability_reports, unwrapped_angles)

        all_bifurcations = saddle_nodes + hopf_points
        all_bifurcations.sort(key=lambda x: x.index)

        return all_bifurcations

    def _interpolate_crossing(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float
    ) -> float:
        if y1 == y0:
            return (x0 + x1) / 2
        t = -y0 / (y1 - y0)
        return x0 + t * (x1 - x0)

def rational_approx(
    value: float,
    max_denominator: int = 20,
    max_numerator: int = 20,
    tolerance: float = 1e-5
) -> Optional[Fraction]:
    """
    Find a rational approximation p/q for a float, subject to size limits.

    The function searches denominators from 1 to max_denominator. For each
    denominator d, it considers the two nearest integers to value * d (floor
    and ceil) as candidate numerators. If a candidate n satisfies
    0 < n ≤ max_numerator and the relative error of n/d is less than tolerance,
    the corresponding Fraction is returned.

    Args:
        value: The number to approximate (any non‑zero float).
        max_denominator: Largest denominator to try (must be ≥ 1).
        max_numerator: Largest numerator allowed (must be ≥ 1).
        tolerance: Relative tolerance for the approximation. The condition is
                   math.isclose(n/d, value, rel_tol=tolerance).

    Returns:
        A Fraction if a suitable approximation is found, otherwise None.
        Returns None immediately if value is None.
    """
    
    if value is None:
        return None
    
    sign = int(np.sign(value))
    value = np.abs(value)
    
    if max_denominator < 1 or max_numerator < 1:
        raise ValueError("max_denominator and max_numerator must be at least 1")

    # Special case exact zero – the only simple representation is 0/1.
    if value == 0.0:
        return Fraction(0, 1)

    for d in range(1, max_denominator + 1):
        product = value * d
        n_floor = int(np.floor(product))
        n_ceil  = int(np.ceil(product))

        for n in (n_floor, n_ceil):
            if n <= 0 or n > max_numerator:
                continue
            approx = n / d
            if np.isclose(approx, value, rtol=tolerance):
                return Fraction(sign * n, d)

    return None