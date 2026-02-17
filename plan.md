# Plan: High Priority Features for pyhbm

This document outlines the implementation plan for four high-priority features to extend the pyhbm library:
1. Time-Domain Validation
2. Stability Analysis (Floquet Exponents)
3. Bifurcation Detection
4. Branch Switching

---

## Feature 1: Time-Domain Validation [DONE]

### Goal
Validate HBM solutions against numerical time integration to verify accuracy of periodic solutions.

### Scientific Background
The Harmonic Balance Method approximates periodic solutions using truncated Fourier series. Time-domain integration provides an independent verification by solving the ODEs directly over multiple periods until transients decay.

### New File
`src/pyhbm/validation/time_domain.py`

### Public API

```python
class TimeDomainValidator:
    """
    Validates HBM solutions against numerical time integration.
    
    Parameters
    ----------
    first_order_ode : FirstOrderODE
        The dynamical system to validate.
    integrator : str, optional
        Integrator name from scipy.integrate. Default is 'RK45'.
    **integrator_kwargs
        Additional arguments passed to the integrator.
    """
    
    def __init__(self, first_order_ode: FirstOrderODE, integrator: str = 'RK45', **integrator_kwargs)
    
    def set_integrator(self, integrator: str, **kwargs) -> None
        """Change the integrator after initialization."""
    
    def validate(
        self, 
        fourier_omega_point: FourierOmegaPoint, 
        omega: float,
        t_span: tuple,
        num_periods: int = 10,
        transient_periods: int = 5
    ) -> ValidationResult:
        """
        Validate an HBM solution against time-domain integration.
        
        Parameters
        ----------
        fourier_omega_point : FourierOmegaPoint
            The HBM solution to validate.
        omega : float
            Angular frequency of the solution.
        t_span : tuple
            Time span for integration (t_start, t_end).
        num_periods : int, optional
            Number of periods to integrate for comparison. Default is 10.
        transient_periods : int, optional
            Number of periods to discard as transients. Default is 5.
        
        Returns
        -------
        ValidationResult
            Object containing error metrics and comparison data.
        """
    
    def compute_state_derivative(self, state: np.ndarray, t: float, omega: float) -> np.ndarray:
        """
        Compute state derivative for ODE integration.
        
        Implements: zdot = omega * f(z, tau) where tau = omega * t
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector of shape (dimension,).
        t : float
            Physical time.
        omega : float
            Angular frequency.
        
        Returns
        -------
        np.ndarray
            State derivative vector.
        """
    
    def compute_error_metrics(
        self, 
        hbm_solution: np.ndarray, 
        td_solution: np.ndarray,
        time_samples: np.ndarray
    ) -> dict:
        """
        Compute error metrics between HBM and time-domain solutions.
        
        Parameters
        ----------
        hbm_solution : np.ndarray
            HBM time series reconstruction.
        td_solution : np.ndarray
            Time-domain integration solution.
        time_samples : np.ndarray
            Time points for comparison.
        
        Returns
        -------
        dict
            Dictionary containing:
            - rms_error: Root mean square error
            - max_error: Maximum absolute error
            - amplitude_error: Relative error in amplitude
            - phase_error: Phase difference in radians
        """
    
    def plot_comparison(
        self, 
        hbm_solution: np.ndarray, 
        td_solution: np.ndarray,
        time_samples: np.ndarray,
        degrees_of_freedom: int = 0,
        show: bool = True,
        **kwargs
    ) -> None:
        """Plot comparison between HBM and time-domain solutions."""
```

### Supported Integrators
- `'RK45'` (default) - Explicit Runge-Kutta 4(5), good for non-stiff
- `'RK23'` - Explicit Runge-Kutta 2(3), lower accuracy, faster
- `'Radau'` - Implicit Radau IIA, for stiff systems
- `'BDF'` - Backward Differentiation Formula, stiff systems
- `'LSODA'` - Automatic stiff/non-stiff detection

### Implementation Notes
1. Convert FourierOmegaPoint to time series using existing `compute_time_series()` method
2. Use the initial condition from the HBM solution's time series
3. Integrate for enough periods to ensure transients have decayed
4. Compare only the last few periods (steady-state)

### ValidationResult Object

```python
@dataclass
class ValidationResult:
    """Container for validation results."""
    
    hbm_time_series: np.ndarray       # HBM reconstructed time series
    td_time_series: np.ndarray        # Time-domain solution
    time_samples: np.ndarray          # Time points
    omega: float                      # Angular frequency
    
    # Error metrics
    rms_error: float                  # Root mean square error
    max_error: float                 # Maximum absolute error
    amplitude_error: float           # Relative amplitude error
    phase_error: float               # Phase error in radians
    
    # Metadata
    num_periods_integrated: int
    transient_periods: int
```

---

## Feature 2: Stability Analysis [DONE]

### Goal
Compute Floquet exponents to determine stability of periodic solutions obtained via HBM.

### Scientific Background
For a periodic solution x(t + T) = x(t) with period T = 2π/ω, the linearized dynamics around the solution are governed by the Jacobian J(t). The Floquet exponents λ are eigenvalues of the average Jacobian J₀:

```
J₀ = (1/T) * ∫₀ᵀ J(t) dt   [0th Fourier harmonic]

Solve: J₀ @ v = λ @ v  →  eigenvalues λ = α + iω

Stability:
- Re(λ) < 0: stable (decaying perturbations)
- Re(λ) > 0: unstable (growing perturbations)
- Re(λ) = 0: neutral (center / bifurcation point)
```

### Optimization
The 0th harmonic of the Jacobian is already computed during Newton-Raphson solving in the HBM. We extract this rather than recomputing.

### New File
`src/pyhbm/stability/stability_analysis.py`

### Public API

```python
class FloquetAnalyzer:
    """
    Computes Floquet exponents for HBM periodic solutions.
    
    Parameters
    ----------
    freq_domain_ode : FrequencyDomainFirstOrderODE
        The frequency-domain representation of the dynamical system.
    """
    
    def __init__(self, freq_domain_ode: FrequencyDomainFirstOrderODE)
    
    def compute_average_jacobian(
        self, 
        fourier_omega_point: FourierOmegaPoint,
        jacobian_RI: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute the average (0th harmonic) Jacobian over one period.
        
        Parameters
        ----------
        fourier_omega_point : FourierOmegaPoint
            The periodic solution.
        jacobian_RI : np.ndarray, optional
            Pre-computed Jacobian from Newton-Raphson in RI format.
            If provided, extracts 0th harmonic directly.
            Otherwise computes via JacobianFourier.
        
        Returns
        -------
        np.ndarray
            Average Jacobian matrix of shape (dimension, dimension).
        """
    
    def compute_floquet_exponents(
        self, 
        fourier_omega_point: FourierOmegaPoint,
        average_jacobian: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute Floquet exponents from average Jacobian.
        
        Solves: J₀ @ v = λ @ v
        
        Parameters
        ----------
        fourier_omega_point : FourierOmegaPoint
            The periodic solution.
        average_jacobian : np.ndarray, optional
            Pre-computed average Jacobian.
        
        Returns
        -------
        np.ndarray
            Complex eigenvalues λ = α + iω of shape (dimension,).
            Real part α: growth/decay rates.
            Imaginary part ω: oscillation frequencies.
        """
    
    @staticmethod
    def classify_stability(exponents: np.ndarray) -> str:
        """
        Classify stability from Floquet exponents.
        
        Parameters
        ----------
        exponents : np.ndarray
            Complex Floquet exponents.
        
        Returns
        -------
        str
            'stable' if all Re(λ) < 0
            'unstable' if any Re(λ) > 0
            'neutral' if any Re(λ) ≈ 0
        """
    
    def analyze(
        self, 
        fourier_omega_point: FourierOmegaPoint,
        jacobian_RI: np.ndarray = None
    ) -> StabilityReport:
        """
        Perform complete stability analysis.
        
        Parameters
        ----------
        fourier_omega_point : FourierOmegaPoint
            The periodic solution to analyze.
        jacobian_RI : np.ndarray, optional
            Pre-computed Jacobian from Newton-Raphson.
        
        Returns
        -------
        StabilityReport
            Complete stability analysis results.
        """
```

### StabilityReport Object

```python
@dataclass
class StabilityReport:
    """Container for stability analysis results."""
    
    # Floquet exponents
    exponents: np.ndarray              # λ = α + iω
    real_parts: np.ndarray             # α (growth/decay rates)
    imaginary_parts: np.ndarray        # ω (oscillation frequencies)
    
    # Classification
    classification: str                 # 'stable', 'unstable', 'neutral'
    is_stable: bool
    
    # Metadata
    omega: float                       # Angular frequency
    period: float                      # Period T = 2π/ω
    dimension: int                     # System dimension
```

### Implementation Notes
1. The Jacobian from Newton-Raphson is in Real-Imaginary format (shape: 2*complex_dim × 2*complex_dim)
2. Extract the 0th harmonic block: this corresponds to the block diagonal element at harmonic index 0
3. Convert back from RI format to complex before computing eigenvalues
4. Use `numpy.linalg.eig` or `scipy.linalg.eig` for eigenvalue computation

---

## Feature 3: Bifurcation Detection

### Goal
Automatically detect saddle-node and Hopf bifurcations during continuation (on-demand).

### Scientific Background

### Saddle-Node (Fold) Bifurcation
- Occurs when two solutions collide and annihilate
- Detected by: dω/dn changes sign along continuation curve
- The ω component of the predictor vector switches sign
- Jacobian becomes singular (determinant → 0)

### Hopf (Neimark-Sacker) Bifurcation
- Complex conjugate Floquet exponents cross the imaginary axis
- Detected by: Re(λ) passes through 0
- At Hopf: λ = ±iω_hopf where ω_hopf is the oscillation frequency of emerging limit cycle
- Relative period = ω_current / ω_hopf (or inverse)

### New File
`src/pyhbm/stability/bifurcation_detection.py`

### Public API

```python
class BifurcationDetector:
    """
    Detects bifurcations in continuation solutions.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize detector.
        
        Parameters
        ----------
        tolerance : float, optional
            Tolerance for detecting zero crossing. Default is 1e-6.
        """
    
    def detect_saddle_nodes(
        self, 
        solution_set: SolutionSet
    ) -> list[SpecialPoint]:
        """
        Detect saddle-node bifurcations.
        
        Monitors dω/dn for sign changes.
        
        Parameters
        ----------
        solution_set : SolutionSet
            Continuation solutions.
        
        Returns
        -------
        list[SpecialPoint]
            List of detected saddle-node points.
        """
    
    def detect_hopf(
        self, 
        solution_set: SolutionSet,
        stability_reports: list[StabilityReport]
    ) -> list[SpecialPoint]:
        """
        Detect Hopf bifurcations.
        
        Monitors real parts of Floquet exponents for zero crossing.
        
        Parameters
        ----------
        solution_set : SolutionSet
            Continuation solutions.
        stability_reports : list[StabilityReport]
            Pre-computed stability reports for each solution.
        
        Returns
        -------
        list[SpecialPoint]
            List of detected Hopf bifurcation points.
        """
    
    def detect_all(
        self, 
        solution_set: SolutionSet,
        stability_reports: list[StabilityReport] = None
    ) -> list[SpecialPoint]:
        """
        Detect all bifurcation types.
        
        Parameters
        ----------
        solution_set : SolutionSet
            Continuation solutions.
        stability_reports : list[StabilityReport], optional
            Pre-computed stability reports. If None, computes on-demand.
        
        Returns
        -------
        list[SpecialPoint]
            List of all detected bifurcation points.
        """
```

### SpecialPoint Object

```python
@dataclass
class SpecialPoint:
    """Represents a bifurcation point."""
    
    # Identification
    type: str                          # 'saddle_node' or 'hopf'
    index: int                        # Index in solution_set
    
    # Location
    omega: float                       # Frequency at bifurcation
    amplitude: float                   # Response amplitude at bifurcation
    refined_omega: float               # Interpolated frequency
    
    # Floquet data (saved at bifurcation point)
    floquet_exponents: np.ndarray      # λ = α + iω at bifurcation
    floquet_exponents_imag: np.ndarray # Im(λ) = oscillation frequencies
    
    # Hopf-specific
    relative_period: float = None      # |new_period / current_period|
    is_commensurate: bool = None       # True if ratio is "nice" rational
    
    def __str__(self) -> str:
        """String representation."""
```

### Commensurability Detection

```python
def is_commensurate(
    ratio: float, 
    tolerance: float = 0.01, 
    max_denominator: int = 10
) -> tuple[bool, fractions.Fraction]:
    """
    Check if a ratio is approximately a simple rational number.
    
    Parameters
    ----------
    ratio : float
        The ratio to check.
    tolerance : float, optional
        Relative tolerance for matching. Default is 0.01 (1%).
    max_denominator : int, optional
        Maximum denominator to check. Default is 10.
    
    Returns
    -------
    tuple[bool, fractions.Fraction]
        (is_commensurate, rational_approximation)
    """
    for p in range(1, max_denominator + 1):
        for q in range(1, max_denominator + 1):
            rational = p / q
            if abs(ratio - rational) < tolerance * rational:
                return True, fractions.Fraction(p, q)
    return False, None
```

### Detection Algorithm Details

#### Saddle-Node Detection
1. Compute dω/dn for each consecutive pair of solutions
2. Monitor sign changes in dω/dn
3. When sign changes, interpolate to find exact ω where dω/dn = 0
4. Verify Jacobian is near-singular at that point

#### Hopf Detection
1. Track real parts of Floquet exponents across solutions
2. Detect when Re(λ) changes sign between consecutive solutions
3. Interpolate to find exact ω where Re(λ) = 0
4. Record Im(λ) = ω_hopf (oscillation frequency at Hopf)
5. Compute relative_period = ω_current / ω_hopf
6. Check commensurability

---

## Feature 4: Branch Switching

### Goal
Automatically continue new solution branches from detected bifurcation points (semi-automatic).

### Scientific Background

### Saddle-Node Branch Switching
- Two solutions collide and disappear
- After the bifurcation, only one branch continues
- Branch switching involves finding the "other" branch by stepping in the opposite direction from the collision point

### Hopf Branch Switching
- At Hopf, a new quasi-periodic branch emerges
- The new periodic orbit has modulation frequency ω_hopf
- If commensurate: new fundamental frequency ω_new = ω_current / relative_period
- If incommensurate: mark and skip (cannot represent with finite harmonics)

### New File
`src/pyhbm/numerical_continuation/branch_switching.py`

### Public API

```python
class BranchSwitcher:
    """
    Performs branch switching from bifurcation points.
    
    Parameters
    ----------
    hbm_solver : HarmonicBalanceMethod
        The HBM solver to use for continuation.
    """
    
    def __init__(self, hbm_solver: HarmonicBalanceMethod)
    
    def switch_from_saddle_node(
        self, 
        bifurcation: SpecialPoint,
        direction: int = 1,
        **continuation_kwargs
    ) -> SolutionSet:
        """
        Switch to the branch emerging from a saddle-node bifurcation.
        
        Parameters
        ----------
        bifurcation : SpecialPoint
            The saddle-node bifurcation point.
        direction : int, optional
            Direction to step: 1 or -1. Default is 1.
        **continuation_kwargs
            Arguments passed to solve_and_continue.
        
        Returns
        -------
        SolutionSet
            The new solution branch.
        """
    
    def switch_from_hopf(
        self, 
        bifurcation: SpecialPoint,
        direction: int = 1,
        **continuation_kwargs
    ) -> SolutionSet | None:
        """
        Switch to the branch emerging from a Hopf bifurcation.
        
        Only proceeds if the bifurcation is commensurate.
        
        Parameters
        ----------
        bifurcation : SpecialPoint
            The Hopf bifurcation point.
        direction : int, optional
            Direction to step: 1 or -1. Default is 1.
        **continuation_kwargs
            Arguments passed to solve_and_continue.
        
        Returns
        -------
        SolutionSet or None
            The new solution branch, or None if incommensurate.
        """
    
    def continue_branch(
        self, 
        initial_guess: FourierOmegaPoint,
        direction: np.ndarray,
        **continuation_kwargs
    ) -> SolutionSet:
        """
        Continue from an initial point in a given direction.
        
        Parameters
        ----------
        initial_guess : FourierOmegaPoint
            Starting point for continuation.
        direction : np.ndarray
            Direction vector in solution space.
        **continuation_kwargs
            Arguments passed to solve_and_continue.
        
        Returns
        -------
        SolutionSet
            The continued solution branch.
        """
    
    @staticmethod
    def find_branch_direction(
        solution_minus: FourierOmegaPoint,
        solution_plus: FourierOmegaPoint
    ) -> np.ndarray:
        """
        Find the tangent direction for a new branch.
        
        Uses difference between two nearby solutions.
        
        Parameters
        ----------
        solution_minus : FourierOmegaPoint
            Solution just before bifurcation.
        solution_plus : FourierOmegaPoint
            Solution just after bifurcation.
        
        Returns
        -------
        np.ndarray
            Tangent direction vector.
        """
```

### Algorithm Details

#### Saddle-Node Algorithm
1. Get the two solutions closest to the bifurcation point
2. Compute difference vector: Δ = solution_plus - solution_minus
3. This gives the tangent direction of the emerging branch
4. Perturb initial guess in that direction
5. Continue using standard HBM continuation

#### Hopf Algorithm
1. Check `bifurcation.is_commensurate`
2. If False: return None (log message)
3. If True:
   - Compute new fundamental ω_new = ω_current / relative_period (or inverse)
   - Create new harmonic set if needed (sidebands)
   - Perturb solution in direction suggested by Floquet eigenvector
   - Continue using standard HBM continuation

### Example Usage

```python
# Run initial continuation
solver = HarmonicBalanceMethod(duffing, harmonics=[1, 3, 5, 7, 9])
solution_set = solver.solve_and_continue(...)

# Detect bifurcations
analyzer = FloquetAnalyzer(solver.freq_domain_ode)
stability_reports = solution_set.analyze_stability(solver.freq_domain_ode)

detector = BifurcationDetector()
bifurcations = detector.detect_all(solution_set, stability_reports)

# Branch switching
switcher = BranchSwitcher(solver)

for bif in bifurcations:
    if bif.type == 'saddle_node':
        new_branch = switcher.switch_from_saddle_node(bif)
    elif bif.type == 'hopf' and bif.is_commensurate:
        new_branch = switcher.switch_from_hopf(bif)
    elif bif.type == 'hopf' and not bif.is_commensurate:
        print(f"Skipping incommensurate Hopf at ω={bif.omega:.4f}")
```

---

## Integration with Existing Code

### Update `SolutionSet` (core.py)

```python
class SolutionSet:
    # ... existing methods ...
    
    def analyze_stability(
        self, 
        freq_domain_ode: FrequencyDomainFirstOrderODE,
        jacobians: list[np.ndarray] = None
    ) -> list[StabilityReport]:
        """
        On-demand stability analysis for all solutions.
        
        Parameters
        ----------
        freq_domain_ode : FrequencyDomainFirstOrderODE
            The frequency domain ODE.
        jacobians : list[np.ndarray], optional
            Pre-computed Jacobians from continuation.
            If None, computes on-demand.
        
        Returns
        -------
        list[StabilityReport]
            Stability report for each solution.
        """
        analyzer = FloquetAnalyzer(freq_domain_ode)
        reports = []
        for i, fourier in enumerate(self.fourier):
            jac = jacobians[i] if jacobians else None
            reports.append(analyzer.analyze(fourier, jac))
        return reports
```

### Update `HarmonicBalanceMethod.solve_and_continue()` (core.py)

Optional parameter to return Jacobians:
```python
def solve_and_continue(
    self, 
    # ... existing parameters ...
    return_jacobians: bool = False,
) -> SolutionSet:
    # ... existing implementation ...
    # Optionally store jacobians and return them
```

### Update Exports (\_\_init\_\_.py)

```python
# Add to existing imports
from .stability import (
    FloquetAnalyzer,
    StabilityReport,
    BifurcationDetector,
    SpecialPoint,
)

from .validation import (
    TimeDomainValidator,
    ValidationResult,
)

from .numerical_continuation import (
    BranchSwitcher,
)
```

---

## Final File Structure

```
src/pyhbm/
├── __init__.py                         # Update exports
├── dynamical_system.py
├── frequency_domain.py
├── core.py                             # Update SolutionSet
├── numerical_continuation/
│   ├── __init__.py
│   ├── corrector_step.py
│   ├── predictor_step.py
│   └── branch_switching.py            # NEW
├── stability/
│   ├── __init__.py                    # NEW
│   ├── stability_analysis.py          # NEW
│   └── bifurcation_detection.py      # NEW
└── validation/
    ├── __init__.py                    # NEW
    └── time_domain.py                 # NEW
```

---

## Implementation Order

### Phase 1: Time-Domain Validation
- Simplest feature, builds understanding
- No dependencies on other new features

### Phase 2: Stability Analysis
- Core feature for bifurcation detection
- Reuses Jacobian from Newton-Raphson

### Phase 3: Bifurcation Detection
- Depends on stability analysis
- On-demand after continuation

### Phase 4: Branch Switching
- Depends on bifurcation detection
- Semi-automatic (user calls it)

---

## Testing Strategy

1. **Unit tests** for each new class/method
2. **Integration tests** using existing examples (Duffing, pendulum, etc.)
3. **Validation** against known analytical results where possible
4. **Compare** with external tools (e.g., AUTO, MatCont, or BifurcationKit)

---

## Backward Compatibility

All new features are additive:
- New classes and methods only
- No changes to existing API signatures
- Optional parameters have sensible defaults

---

## Dependencies

No new external dependencies required. Uses:
- `numpy`
- `scipy` (for integration and eigenvalue computation)
- `matplotlib` (for plotting)
- `fractions` (standard library, for rational detection)
