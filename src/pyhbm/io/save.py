import pickle

import numpy as np
from numpy import array

from ..frequency_domain import Fourier


def save_solution_set(solution_set, path, 
                      harmonic_amplitude=True, 
                      amplitude=True, 
                      angular_frequency=True,
                      fourier_coefficients=False, 
                      time_series=False, 
                      adimensional_time_samples=False,
                      iterations=False, 
                      step_length=False,
                      MATLAB_compatible=False):
    
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

    if MATLAB_compatible:
        from scipy.io import savemat
        savemat(path, solution_data)
    else:
        with open(path, 'wb') as handle:
            pickle.dump(solution_data, handle)
