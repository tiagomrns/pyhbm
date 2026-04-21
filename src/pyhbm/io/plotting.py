import numpy as np
from numpy import array
from numpy.linalg import norm
from matplotlib import pyplot as plt

from ..frequency_domain import Fourier
from ..stability import FloquetAnalyzer


def _segment_by_stability(solution_set, time_domain_ode, stability_reports=None):

    if stability_reports is not None:
        stabilities = [report.is_stable for report in stability_reports]
    else:
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


def _plot_bifurcation_points(bifurcations, omega_array, amplitude_array, reference_omega=None, **kwargs):
    """
    Plot bifurcation points on the FRF curve.
    
    Uses the index stored in each bifurcation point to get the correct y-value
    from the user's computed amplitude array.
    """
    marker_size = kwargs.pop('markersize', 100)
    marker_edge_width = kwargs.pop('markeredgewidth', 1.5)
    
    saddle_nodes = [b for b in bifurcations if b.type == 'Saddle-Node']
    hopf_points = [b for b in bifurcations if b.type == 'Hopf']
    
    def get_coords(bif_points):
        """Get omega and amplitude arrays using indices."""
        if not bif_points:
            return np.array([]), np.array([])
        indices = np.array([b.index for b in bif_points])
        omegas = np.array([omega_array[i] for i in indices])
        amps = np.array([amplitude_array[i] for i in indices])
        if reference_omega is not None:
            omegas = omegas / reference_omega
        return omegas, amps
    
    # Plot saddle-nodes (blue circles, edge only)
    sn_omegas, sn_amps = get_coords(saddle_nodes)
    if len(sn_omegas) > 0:
        plt.scatter(sn_omegas, sn_amps, 
                   marker='o', c='none', edgecolors='blue', 
                   s=marker_size, linewidths=marker_edge_width,
                   label='Saddle-Node', zorder=10)
    
    # Plot Hopf points (green diamonds, filled)
    hf_omegas, hf_amps = get_coords(hopf_points)
    if len(hf_omegas) > 0:
        plt.scatter(hf_omegas, hf_amps, 
                   marker='x', c='green', 
                   s=marker_size, linewidths=marker_edge_width,
                   label='Hopf', zorder=10)
    
    if len(sn_omegas) > 0 or len(hf_omegas) > 0:
        plt.legend(loc='best')


def plot_FRF(solution_set, degrees_of_freedom, harmonic=None, reference_omega=None, 
             yscale='linear', xscale='linear', show=True, 
             time_domain_ode=None, stability_reports=None, bifurcations=None, **kwargs):
    
    # Compute harmonic amplitude
    harmonic_amplitude = None
    
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

    if stability_reports is not None:
        if time_domain_ode is None:
            raise ValueError("time_domain_ode required for stability coloring")
        segments, statuses = _segment_by_stability(solution_set, time_domain_ode, stability_reports)
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

    # Plot bifurcation points
    if bifurcations is not None:
        _plot_bifurcation_points(
            bifurcations, 
            solution_set.omega, 
            harmonic_amplitude,
            reference_omega,
            **kwargs
        )

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
