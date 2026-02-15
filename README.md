# pyhbm

Python Harmonic Balance Method library for nonlinear dynamics analysis.

## Overview

pyhbm provides tools for solving nonlinear dynamical systems using the Harmonic Balance Method (HBM) with numerical continuation. It supports:

- Frequency domain analysis with arbitrary harmonic expansions
- Numerical continuation for tracking solutions across parameter ranges
- Both autonomous and non-autonomous systems
- Real and complex Fourier representations

## Installation

### Requirements
- Python >= 3.9
- numpy >= 1.20
- scipy >= 1.8
- matplotlib >= 3.5

### From PyPI
```bash
python -m pip install pyhbm
```

### For Development
```bash
python -m pip install -e .
```

### Install Dependencies Only
```bash
python -m pip install numpy scipy matplotlib
```

## Quick Start

```python
import numpy as np
from pyhbm import HarmonicBalanceMethod, FourierOmegaPoint, FirstOrderODE

class DuffingForced(FirstOrderODE):
    def __init__(self, c=0.01, k=1.0, beta=1.0, P=1.0):
        self.c = c
        self.k = k
        self.beta = beta
        self.P = P
        self.linear_coefficient = np.array([[0.0, 1.0], [-k, -c]])
        self.dimension = 2
        self.polynomial_degree = 3
    
    def external_term(self, adimensional_time):
        zeros = np.zeros_like(adimensional_time)
        return np.array([[zeros, self.P * np.cos(adimensional_time)]]).transpose()
    
    def linear_term(self, state):
        return self.linear_coefficient @ state
    
    def nonlinear_term(self, state, adimensional_time):
        u = state[..., 0:1, :]
        zeros = np.zeros_like(u)
        fnl = -self.beta * np.power(u, 3)
        return np.concatenate((zeros, fnl), axis=-2)
    
    def jacobian_nonlinear_term(self, state, adimensional_time):
        u = state[..., 0:1, :]
        zeros = np.zeros_like(u)
        dfnldx = -3 * self.beta * np.power(u, 2)
        jacobian1 = np.concatenate((zeros, zeros), axis=-1)
        jacobian2 = np.concatenate((dfnldx, zeros), axis=-1)
        return np.concatenate((jacobian1, jacobian2), axis=-2)

duffing = DuffingForced(c=0.009, k=1.0, beta=1.0, P=1.0)
solver = HarmonicBalanceMethod(
    first_order_ode=duffing, 
    harmonics=[1, 3, 5, 7, 9],
)

initial_omega = 0.0
first_harmonic = np.array([[1], [1j * initial_omega]])
static_amplitude = duffing.P / duffing.k
initial_guess = FourierOmegaPoint.new_from_first_harmonic(
    first_harmonic * static_amplitude, 
    omega=initial_omega
)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(
    first_harmonic, 
    omega=1
)

solution_set = solver.solve_and_continue(
    initial_guess=initial_guess,
    initial_reference_direction=initial_reference_direction,
    maximum_number_of_solutions=1000,
    angular_frequency_range=[0.0, 10],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs={
        "base": 2,
        "initial_step_length": 0.1,
        "maximum_step_length": 2.0,
    }
)

solution_set.plot_FRF(degrees_of_freedom=0, xscale='log', yscale='log')
```

## Examples

The `examples/` directory contains several working examples:

| Example | Description |
|---------|-------------|
| `duffing_forced_nonautonomous/` | Forced Duffing oscillator |
| `duffing_conservative_autonomous/` | Conservative Duffing (autonomous) |
| `pendulum_forced_nonautonomous/` | Forced pendulum |
| `2dof_duffing/` | 2-DOF Duffing system |
| `linear_frequency_response/` | Linear system frequency response |
| `arch_beam_ssm/` | Arch beam SSM analysis |

Run an example:
```bash
cd examples/duffing_forced_nonautonomous
python main.py
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `HarmonicBalanceMethod` | Main solver class that combines Harmonic Balance Method with numerical continuation for tracking solution families across parameter ranges |
| `SolutionSet` | Container that stores families of solutions obtained from continuation, with methods for plotting and analysis |
| `FourierOmegaPoint` | Represents the solution state in frequency domain with Fourier coefficients and angular frequency |
| `Fourier` | Base class for Fourier coefficient representation (used internally) |

### Frequency Domain

| Class | Description |
|-------|-------------|
| `Fourier` | Base Fourier representation class for handling harmonic coefficients |
| `Fourier_Real` | Real-valued Fourier coefficients for systems with symmetric solutions |
| `Fourier_Complex` | Complex-valued Fourier coefficients for general periodic solutions |
| `FourierOmegaPoint` | Combined Fourier representation with angular frequency parameter |
| `FirstOrderODE` | Base class for defining dynamical systems; requires implementing `external_term()`, `linear_term()`, `nonlinear_term()`, and optionally `jacobian_nonlinear_term()` |
| `FrequencyDomainFirstOrderODE` | Wrapper that converts time-domain ODE to frequency domain for HBM solving |

### Numerical Continuation

| Class | Description |
|-------|-------------|
| `NewtonRaphson` | Newton-Raphson iterative solver used as corrector to find solutions on the continuation curve |
| `Predictor` | Base predictor class defining the interface for continuation predictors |
| `TangentPredictorOne` | First-order tangent predictor using linear approximation of the continuation curve |
| `TangentPredictorTwo` | Second-order predictor using quadratic approximation for better accuracy |
| `TangentPredictorRobust` | Robust tangent predictor with adaptive step sizing for difficult regions |
| `StepLengthAdaptation` | Base class for adaptive step length control during continuation |
| `ExponentialAdaptation` | Exponential step size adaptation adjusting step length based on iteration count |
| `BiExponentialAdaptation` | Bi-exponential adaptation with separate growth and decay rates |
| `OrthogonalParameterization` | Orthogonal correction method for improved stability in multi-parameter continuation |

## License

MIT
