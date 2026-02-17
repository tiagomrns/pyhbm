import numpy as np
from numpy import array
from numpy.linalg import norm
from matplotlib import pyplot as plt

from ..frequency_domain import Fourier
from ..stability import FloquetAnalyzer


def _segment_by_stability(solution_set, time_domain_ode):

    analyzer = FloquetAnalyzer(time_domain_ode = time_domain_ode)
    adimensional_time_samples = Fourier.adimensional_time_samples

    stabilities = []
    for i, fourier in enumerate(solution_set.fourier):
        omega = solution_set.omega[i]
        report = analyzer.analyze(fourier.time_series, adimensional_time_samples, omega)
        stabilities.append(report.is_stable)

    segments = []
    current_status = stabilities[0]
    current_indices = [0]

    for i in range(1, len(stabilities)):
        current_indices.append(i)
        if stabilities[i] != current_status:
            segments.append((current_indices, current_status))
            current_status = stabilities[i]
            current_indices = [i]   
    segments.append((current_indices, current_status))

    return [idx for idx, _ in segments], [status for _, status in segments]


def plot_FRF(solution_set, degrees_of_freedom, harmonic=None, reference_omega=None, 
             yscale='linear', xscale='linear', show=True, stability=False, 
             time_domain_ode=None, **kwargs):
    
    for dof in array([degrees_of_freedom]).ravel():
        if harmonic is None:
            harmonic_amplitude = norm(
                array([fourier.coefficients for fourier in solution_set.fourier])[:, :, dof, 0], 
                axis=1
            ) * 2 / Fourier.number_of_time_samples
            plt.ylabel(r"$||\mathbf{Q}||_{DoF=%d}$" % (dof))
        else:
            index = list(Fourier.harmonics).index(harmonic)
            harmonic_amplitude = abs(
                array([fourier.coefficients for fourier in solution_set.fourier])[:, index, dof, 0]
            ) * 2 / Fourier.number_of_time_samples
            plt.ylabel(r"$|Q_{%d, %d}|$" % (harmonic, dof))

    if stability:
        if time_domain_ode is None:
            raise ValueError("stability analysis requires time_domain_ode to be provided")
        segments, statuses = _segment_by_stability(solution_set, time_domain_ode)
        for indices, is_stable in zip(segments, statuses):
            if reference_omega is None:
                omegas = array(solution_set.omega)[indices]
            else:
                omegas = array(solution_set.omega)[indices] / reference_omega
            amps = harmonic_amplitude[indices]
            color = 'black' if is_stable else 'red'
            plt.plot(omegas, amps, color=color, **kwargs)
    else:
        if reference_omega is None:
            plt.plot(solution_set.omega, harmonic_amplitude, **kwargs)
        else:
            omega = array(solution_set.omega) / reference_omega
            plt.plot(omega, harmonic_amplitude, **kwargs)

    if reference_omega is None:
        plt.xlabel(r"$\omega$")
    else:
        plt.xlabel(r"$\omega/\omega_0$")

    if yscale == 'log':
        plt.yscale('log')
    if xscale == 'log':
        plt.xscale('log')

    if show:
        plt.show()
