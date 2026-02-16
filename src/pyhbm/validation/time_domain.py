from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
from scipy.fft import fft

from ..dynamical_system import FirstOrderODE


SUPPORTED_INTEGRATORS = ['RK45', 'RK23', 'Radau', 'BDF', 'LSODA']

DEFAULT_AMPLITUDE_THRESHOLD = 1e-10


def _validate_time_series(time_series: np.ndarray) -> None:
    if time_series.ndim == 0:
        raise ValueError("time_series must be at least 1D")
    if time_series.ndim > 3:
        raise ValueError(f"time_series must have at most 3 axes, got {time_series.ndim}")


@dataclass(frozen=True)
class ValidationResult:
    ref_time_series: np.ndarray
    td_time_series: np.ndarray
    time_samples: np.ndarray
    omega: float

    relative_rms_error: float
    relative_max_error: float
    phase_error: float

    multiplier_sampling_rate: int
    num_degrees_of_freedom: int


class TimeDomainValidator:
    def __init__(
        self,
        first_order_ode: FirstOrderODE,
        integrator: str = 'RK45',
        amplitude_threshold: float = DEFAULT_AMPLITUDE_THRESHOLD,
        **integrator_kwargs: Any
    ) -> None:
        """Initialize the TimeDomainValidator.

        Args:
            first_order_ode: The first-order ODE system to validate.
            integrator: The ODE integrator to use. Supported: 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA'.
            amplitude_threshold: Minimum amplitude threshold for relative error computation.
            **integrator_kwargs: Additional keyword arguments passed to the ODE solver.
        """
        if integrator not in SUPPORTED_INTEGRATORS:
            raise ValueError(
                f"Unknown integrator '{integrator}'. "
                f"Supported: {SUPPORTED_INTEGRATORS}"
            )

        self.first_order_ode = first_order_ode
        self.integrator = integrator
        self.amplitude_threshold = amplitude_threshold
        self.integrator_kwargs = integrator_kwargs

    def set_integrator(self, integrator: str, **kwargs: Any) -> None:
        """Set the ODE integrator and its options.

        Args:
            integrator: The ODE integrator to use.
            **kwargs: Additional keyword arguments passed to the ODE solver.
        """
        if integrator not in SUPPORTED_INTEGRATORS:
            raise ValueError(
                f"Unknown integrator '{integrator}'. "
                f"Supported: {SUPPORTED_INTEGRATORS}"
            )
        self.integrator = integrator
        self.integrator_kwargs = kwargs

    def ode(
        self,
        state: np.ndarray,
        adim_time: float,
        omega: float
    ) -> np.ndarray:
        """Compute the ODE right-hand side for the time domain solution.

        Args:
            state: Current state vector.
            adim_time: Dimensionless time.
            omega: Angular frequency.

        Returns:
            Time derivative of the state.
        """
        linear = self.first_order_ode.linear_term(state)
        nonlinear = self.first_order_ode.nonlinear_term(state, adim_time)
        external = self.first_order_ode.external_term(adim_time)

        return (linear + nonlinear + external) / omega

    def validate(
        self,
        time_series: np.ndarray,
        omega: float,
        multiplier_sampling_rate: int = 1,
    ) -> ValidationResult:
        """Validate the HBM solution against time-domain integration.

        Args:
            time_series: Reference time series from HBM, shape (n_points, n_dof) or (n_points,).
            omega: Angular frequency of the solution.
            multiplier_sampling_rate: Sampling rate multiplier for time-domain integration.

        Returns:
            ValidationResult containing the comparison and error metrics.

        Raises:
            ValueError: If omega <= 0 or multiplier_sampling_rate < 1.
        """
        if omega <= 0:
            raise ValueError(f"omega must be positive, got {omega}")
        if not isinstance(multiplier_sampling_rate, (int, np.integer)) or multiplier_sampling_rate < 1:
            raise ValueError(f"multiplier_sampling_rate must be a positive integer, got {multiplier_sampling_rate}")
        _validate_time_series(time_series)

        ref_time_series = time_series.copy()

        if ref_time_series.ndim == 3:
            ref_time_series = ref_time_series.reshape(
                ref_time_series.shape[0],
                ref_time_series.shape[1]
            )

        initial_state = ref_time_series[0, :]

        num_time_points_per_period = multiplier_sampling_rate * len(ref_time_series)

        tau_end = 2 * np.pi

        adimensional_time_samples = np.linspace(0, tau_end, num_time_points_per_period, endpoint=False)

        sol = solve_ivp(
            fun=lambda tau, y: self.ode(y, tau, omega),
            t_span=(0, tau_end),
            y0=initial_state,
            method=self.integrator,
            t_eval=adimensional_time_samples,
            vectorized=True,
            **self.integrator_kwargs
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        td_time_series = sol.y.T

        error_metrics = self.compute_error_metrics(
            ref_time_series,
            td_time_series[::multiplier_sampling_rate]
        )

        num_dof = ref_time_series.shape[1]

        return ValidationResult(
            ref_time_series=ref_time_series,
            td_time_series=td_time_series,
            time_samples=adimensional_time_samples / omega,
            omega=omega,
            relative_rms_error=error_metrics['relative_rms_error'],
            relative_max_error=error_metrics['relative_max_error'],
            phase_error=error_metrics['phase_error'],
            multiplier_sampling_rate=multiplier_sampling_rate,
            num_degrees_of_freedom=num_dof
        )

    def compute_error_metrics(
        self,
        ref_solution: np.ndarray,
        td_solution: np.ndarray,
    ) -> dict[str, float]:
        """Compute error metrics between reference and time-domain solutions.

        Uses cross-correlation to find optimal alignment before computing errors.

        Args:
            ref_solution: Reference time series (e.g., HBM solution).
            td_solution: Time-domain solution to compare against.

        Returns:
            Dictionary with 'relative_rms_error', 'relative_max_error', and 'phase_error'.
        """
        n_time = ref_solution.shape[0]
        n_dof = ref_solution.shape[1]

        best_shift = 0
        best_error = float('inf')

        if n_dof == 1:
            ref_flat = ref_solution[:, 0]
            td_flat = td_solution[:, 0]

            correlation = np.correlate(ref_flat - ref_flat.mean(), td_flat - td_flat.mean(), mode='full')
            correlation = correlation[n_time - 1: 2 * n_time - 1]
            best_shift = np.argmax(correlation)
            best_error = np.mean((ref_flat - np.roll(td_flat, best_shift)) ** 2)
        else:
            for shift in range(n_time):
                shifted_td = np.roll(td_solution, shift, axis=0)
                error = np.mean((ref_solution - shifted_td) ** 2)
                if error < best_error:
                    best_error = error
                    best_shift = shift

        td_aligned = np.roll(td_solution, best_shift, axis=0)
        max_abs_amplitude = max(np.max(np.abs(td_aligned)), np.max(np.abs(ref_solution)))

        difference = ref_solution - td_aligned

        if max_abs_amplitude > self.amplitude_threshold:
            relative_rms_error = np.sqrt(np.mean(difference ** 2)) / max_abs_amplitude
            relative_max_error = np.max(np.abs(difference)) / max_abs_amplitude
        else:
            relative_rms_error = 0.0
            relative_max_error = 0.0

        ref_fft = fft(ref_solution, axis=0)
        td_fft = fft(td_aligned, axis=0)

        ref_phase = np.angle(ref_fft[1, :])  # type: ignore[index]
        td_phase = np.angle(td_fft[1, :])  # type: ignore[index]

        phase_error = np.abs(ref_phase - td_phase)
        phase_error = np.where(phase_error > np.pi, 2 * np.pi - phase_error, phase_error)
        phase_error = float(np.mean(phase_error))

        return {
            'relative_rms_error': relative_rms_error,
            'relative_max_error': relative_max_error,
            'phase_error': phase_error
        }

    def plot_comparison(
        self,
        validation_result: ValidationResult,
        degrees_of_freedom: int = 0,
        show: bool = True,
        **kwargs: Any
    ) -> Optional[Figure]:
        """Plot comparison between HBM and time-domain solutions.

        Args:
            validation_result: The validation result to plot.
            degrees_of_freedom: Index of the degree of freedom to plot.
            show: Whether to display the plot.
            **kwargs: Additional keyword arguments passed to matplotlib plot.

        Returns:
            The matplotlib Figure object, or None if show=True.
        """
        ref_ts = validation_result.ref_time_series
        td_ts = validation_result.td_time_series

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        ref_sampled_time = validation_result.time_samples[::validation_result.multiplier_sampling_rate]

        axes[0].plot(
            ref_sampled_time,
            ref_ts[:, degrees_of_freedom],
            'b-',
            label='HBM',
            **kwargs
        )
        axes[0].plot(
            validation_result.time_samples,
            td_ts[:, degrees_of_freedom],
            'r--',
            label='Time Domain',
            **kwargs
        )
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel(f'DoF {degrees_of_freedom}')
        axes[0].legend()
        axes[0].set_title('Steady-State Comparison')
        axes[0].grid(True)

        axes[1].plot(
            ref_sampled_time,
            ref_ts[:, degrees_of_freedom] - td_ts[::validation_result.multiplier_sampling_rate, degrees_of_freedom],
            'ko',
            label='Error',
            **kwargs
        )
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Relative Error')
        axes[1].legend()
        axes[1].set_title(
            f'Relative Error (RMS: {validation_result.relative_rms_error:.2e}, '
            f'Relative Max: {validation_result.relative_max_error:.2e})'
        )
        axes[1].grid(True)

        plt.tight_layout()

        if show:
            plt.show()
            return None

        return fig
