import numpy as np
from numpy import vdot, sign
from numpy.linalg import norm
from scipy.linalg import null_space

class Predictor(object):
    def compute_prediction():
        pass

class TangentPredictor(Predictor):
    def compute_predictor_vector(step_length, jacobian: np.ndarray, reference_predictor_vector: np.ndarray) -> np.ndarray:
        predictor_vector: np.ndarray = null_space(jacobian)
        # TODO: select between multiple possible predictor vectors
        # normalize predictor_vector
        predictor_vector /= norm(predictor_vector)
        # align predictor_vector with reference
        predictor_vector *= sign(vdot(reference_predictor_vector, predictor_vector))
        # scale to match step length
        return predictor_vector * step_length

class StepLengthAdaptation(object):
    def update_step_length():
        pass

class ExponentialAdaptation(StepLengthAdaptation):
    def __init__(self, base, user_step_length, max_step_length, min_step_length, goal_number_of_iterations):

        assert base > 1, "base must be greater than 1"
        self.base = base

        self.user_step_length = user_step_length
        self.max_step_length = max_step_length
        self.min_step_length = min_step_length
        self.goal_number_of_iterations = goal_number_of_iterations
        
        self.step_length = user_step_length

    def update_step_length(self, iterations):
        new_step_length = self.step_length * (self.base**(self.goal_number_of_iterations - iterations))
        self.step_length = min(max(new_step_length, self.min_step_length), self.max_step_length)