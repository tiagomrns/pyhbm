import pickle
import json

import numpy as np
from numpy import array

from ..frequency_domain import Fourier


def _save_bifurcation_data(bifurcations: list) -> list:
    """
    Convert bifurcations to serializable dictionary format.
    
    Parameters
    ----------
    bifurcations : list
        List of SpecialPoint objects.
    
    Returns
    -------
    list
        List of dictionaries with bifurcation data.
    """
    data = []
    for bif in bifurcations:
        bif_data = {
            'type': bif.type,
            'index': int(bif.index),
            'omega': float(bif.omega),
        }
        if bif.refined_omega is not None:
            bif_data['refined_omega'] = float(bif.refined_omega)
        
        if bif.type == 'hopf':
            if bif.relative_period is not None:
                bif_data['relative_period'] = float(bif.relative_period)
                bif_data['rational_approx'] = bif.rational_approx
            if bif.hopf_frequency is not None:
                bif_data['hopf_frequency'] = float(bif.hopf_frequency)
            if bif.hopf_multiplier_magnitude is not None:
                bif_data['hopf_multiplier_magnitude'] = float(bif.hopf_multiplier_magnitude)
            if bif.multipliers is not None:
                bif_data['multipliers'] = [
                    {'real': float(m.real), 'imag': float(m.imag)} 
                    for m in bif.multipliers
                ]
        
        data.append(bif_data)
    
    return data


def save_solution_set(solution_set, path, 
                      harmonic_amplitude=True, 
                      amplitude=True, 
                      angular_frequency=True,
                      fourier_coefficients=False, 
                      time_series=False, 
                      adimensional_time_samples=False,
                      iterations=False, 
                      step_length=False,
                      bifurcations=False,
                      MATLAB_compatible=False,
                      freq_domain_ode=None):
    
    solution_data = {}

    if harmonic_amplitude:
        solution_data["harmonic_amplitude"] = \
            abs(array([fourier.coefficients[..., 0] for fourier in solution_set.fourier])) * 2 / Fourier.number_of_time_samples
    
    if amplitude:
        solution_data["amplitude"] = array([np.max(abs(fourier.time_series), axis=0) for fourier in solution_set.fourier])

    if angular_frequency:
        solution_data["angular_frequency"] = solution_set.omega.copy()

    if fourier_coefficients:
        solution_data["fourier_coefficients"] = array([fourier.coefficients.copy() for fourier in solution_set.fourier])

    if time_series:
        solution_data["time_series"] = array([fourier.time_series.copy() for fourier in solution_set.fourier])

    if adimensional_time_samples:
        solution_data["adimensional_time_samples"] = Fourier.adimensional_time_samples

    if iterations:
        solution_data["iterations"] = solution_set.iterations.copy()

    if step_length:
        solution_data["step_length"] = solution_set.step_length.copy()

    if bifurcations:
        if freq_domain_ode is None:
            raise ValueError("freq_domain_ode is required when bifurcations=True")
        
        from ..stability import BifurcationDetector, FloquetAnalyzer
        
        # Compute stability reports
        analyzer = FloquetAnalyzer(freq_domain_ode.ode)
        stability_reports = []
        for fourier in solution_set.fourier:
            if fourier.time_series is None:
                fourier.compute_time_series()
            report = analyzer.analyze(
                fourier.time_series, 
                Fourier.adimensional_time_samples, 
                solution_set.omega[len(stability_reports)]
            )
            stability_reports.append(report)
        
        # Detect bifurcations
        detector = BifurcationDetector()
        bifurcations_data = detector.detect_all(solution_set, stability_reports)
        
        # Save to solution_data
        solution_data['bifurcations'] = _save_bifurcation_data(bifurcations_data)

    if MATLAB_compatible:
        from scipy.io import savemat
        savemat(path, solution_data)
    else:
        with open(path, 'wb') as handle:
            pickle.dump(solution_data, handle)
