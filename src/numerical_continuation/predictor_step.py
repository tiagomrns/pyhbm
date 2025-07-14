import numpy as np
from numpy import vdot, sign
from numpy.linalg import norm
from scipy.linalg import null_space

class Predictor(object):
    def compute_predictor_vector():
        pass

class TangentPredictorRobust(Predictor):
    """
    Robust tangent predictor
    The dimension of the kernel is verified
    It is not the fastest if the dimension of the kernel is known apriori
    """
    @staticmethod
    def compute_predictor_vector(step_length: float, 
                                 jacobian: np.ndarray, 
                                 reference_direction: np.ndarray, 
                                 remove_directions=np.array([[]])) -> np.ndarray:
        
        predictor_vector: np.ndarray = null_space(jacobian)
        dimension_of_kernel: int = predictor_vector.shape[1]
        
        if dimension_of_kernel != 1:
            
            if dimension_of_kernel == 0:
                print(f"TangentPredictorRobust: could not find any predictor directions")
                return None
            
            predictor_vector = TangentPredictorRobust.filter_directions(predictor_vector, 
                                                                        dimension_of_kernel, 
                                                                        remove_directions)
        
        if predictor_vector is None:
            return None
            
        # normalize predictor_vector
        predictor_vector /= norm(predictor_vector)
        # align predictor_vector with reference
        predictor_vector *= sign(vdot(reference_direction, predictor_vector))
        # scale to match step length
        return predictor_vector * step_length
    
    @staticmethod
    def filter_directions(predictor_vector: np.ndarray, 
                          dimension_of_kernel: int, 
                          remove_directions: np.ndarray):
        
        remove_dimensions: int = remove_directions.shape[1]   
         
        if dimension_of_kernel > remove_dimensions + 1:
            print(f"TangentPredictorRobust: encoutered {dimension_of_kernel} possible predictor directions and could only remove {remove_dimensions}")
            return None
            
        alignment_to_remove: np.ndarray = remove_directions.T @ predictor_vector
        coordinates_filter: np.ndarray = null_space(alignment_to_remove)
        
        if coordinates_filter.shape[1] == 1:
            return predictor_vector @ coordinates_filter
        
        print(f"TangentPredictorRobust: encoutered {dimension_of_kernel} possible predictor directions. \
                After filering, {coordinates_filter.shape[1]} directions remained because the remove directions were not independent")
        
        return None
        
class TangentPredictorOne(Predictor):
    """
    Less robust than TangentPredictorRobust 
    The dimension of the kernel is always assumed to be 1
    It is designed for non-autonomous ODEs
    Use at your own risk
    """
    autonomous = False
    @staticmethod
    def compute_predictor_vector(step_length: float, 
                                 jacobian: np.ndarray, 
                                 reference_direction: np.ndarray) -> np.ndarray:
        
        predictor_vector: np.ndarray = null_space(jacobian)
        # normalize predictor_vector
        predictor_vector /= norm(predictor_vector)
        # align predictor_vector with reference
        predictor_vector *= sign(vdot(reference_direction, predictor_vector))
        # scale to match step length
        return predictor_vector * step_length
        
class TangentPredictorTwo(Predictor):
    """
    Less robust than TangentPredictorRobust 
    The dimension of the kernel is always assumed to be 2
    It is designed for autonomous ODEs
    Use at your own risk
    """
    autonomous = True
    @staticmethod
    def compute_predictor_vector(step_length: float, 
                                 jacobian: np.ndarray, 
                                 reference_direction: np.ndarray, 
                                 remove_direction: np.ndarray,
                                 rcond: float = None) -> np.ndarray:
        
        predictor_vector: np.ndarray = null_space(jacobian, rcond=rcond)
        dimension_of_kernel: int = predictor_vector.shape[1]
        assert dimension_of_kernel == 2, f"TangentPredictorTwo: for rcond={rcond}, dimension_of_kernel={dimension_of_kernel}"

        # remove one direction
        alignment_to_remove: np.ndarray = remove_direction.T @ predictor_vector
        predictor_vector = predictor_vector @ np.array([-alignment_to_remove[:,1], alignment_to_remove[:,0]])
            
        # normalize predictor_vector
        predictor_vector /= norm(predictor_vector)
        # align predictor_vector with reference
        predictor_vector *= sign(vdot(reference_direction, predictor_vector))
        # scale to match step length
        return predictor_vector * step_length

#%%    

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