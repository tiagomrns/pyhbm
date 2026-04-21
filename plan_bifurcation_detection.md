# Plan: Feature 3 - Bifurcation Detection

## Overview

Feature 3 implements automatic detection of saddle-node (fold) and Hopf bifurcations during continuation. This feature analyzes a `SolutionSet` and identifies points where the qualitative behavior of the system changes.

## Current State of the Codebase

### Existing Components (from Features 1 & 2)

1. **`StabilityReport`** (in `stability/stability_analysis.py`):
   - Stores `multipliers` (Floquet multipliers via monodromy matrix)
   - Has `is_stable` boolean
   - Uses `FloquetAnalyzer` which computes multipliers, not exponents

2. **`FloquetAnalyzer`**:
   - Computes monodromy matrix by propagating variational equation
   - Uses `time_domain_ode` (FirstOrderODE) for Jacobian computation
   - Method `analyze(time_series, adimensional_time_samples, omega)` returns StabilityReport

3. **`SolutionSet`** (in `core.py`):
   - Has `analyze_stability(freq_domain_ode)` method returning `list[StabilityReport]`
   - Stores: `fourier`, `omega`, `iterations`, `step_length` as lists

### Key Implementation Notes

1. **Multipliers vs Exponents**: The current implementation uses Floquet multipliers (via monodromy matrix). The relationship to exponents:
   - μ = exp(λ·T) where T = 2π/ω is the period
   - Stable: |μ| < 1, Unstable: |μ| > 1, Hopf: |μ| = 1

2. **Data Available**:
   - Each solution in SolutionSet has: `fourier` (Fourier coefficients), `omega`, `iterations`, `step_length`
   - Stability analysis provides: `multipliers` (complex array)

---

## Implementation Details

### New File Structure

```
src/pyhbm/stability/
├── __init__.py                    # UPDATE: add BifurcationDetector, SpecialPoint
├── stability_analysis.py          # EXISTING (Feature 2)
└── bifurcation_detection.py      # NEW (Feature 3)
```

### Files to Modify

1. `src/pyhbm/stability/__init__.py` - Add exports
2. `src/pyhbm/core.py` - Update `SolutionSet` with bifurcation detection convenience method

---

## Class: SpecialPoint

Represents a detected bifurcation point.

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

@dataclass
class SpecialPoint:
    """Represents a detected bifurcation point."""
    
    # Identification
    type: str                          # 'saddle_node' or 'hopf'
    index: int                        # Index in solution_set
    
    # Location
    omega: float                       # Frequency at bifurcation
    amplitude: float                   # Response amplitude (max of time series)
    refined_omega: Optional[float]     # Interpolated frequency (if refined)
    
    # Multiplier data (saved at bifurcation point)
    multipliers: NDArray[np.complexfloating]  # Floquet multipliers at bifurcation
    
    # Hopf-specific
f_multiplier_magnitude: Optional    hop[float] = None  # |μ| at Hopf (≈1)
    hopf_frequency: Optional[float] = None             # Im(μ) related frequency
    relative_period: Optional[float] = None            # |ω/ω_hopf|
    is_commensurate: Optional[bool] = None              # True if ratio is "nice" rational
    
    def __str__(self) -> str:
        """String representation."""
        base = f"{self.type} at ω={self.omega:.4f}, index={self.index}"
        if self.type == 'hopf' and self.is_commensurate is not None:
            base += f", commensurate={self.is_commensurate}"
        return base
```

---

## Class: BifurcationDetector

Detects bifurcations in continuation solutions.

```python
import numpy as np
from numpy.typing import NDArray
from fractions import Fraction
from typing import Optional

from .stability_analysis import FloquetAnalyzer, StabilityReport
from ..core import SolutionSet


class BifurcationDetector:
    """
    Detects bifurcations in continuation solutions.
    
    Parameters
    ----------
    tolerance : float, optional
        Tolerance for detecting zero crossings and commensurability.
        Default is 1e-6 for zero crossings, 0.01 (1%) for commensurability.
    """
    
    def __init__(self, tolerance: float = 1e-6, commensurability_tolerance: float = 0.01):
        """
        Initialize detector.
        
        Parameters
        ----------
        tolerance : float, optional
            Tolerance for detecting zero crossing. Default is 1e-6.
        commensurability_tolerance : float, optional
            Relative tolerance for commensurability detection. Default is 0.01 (1%).
        """
        self.tolerance = tolerance
        self.commensurability_tolerance = commensurability_tolerance
    
    # =========================================================================
    # Saddle-Node (Fold) Detection
    # =========================================================================
    
    def detect_saddle_nodes(
        self, 
        solution_set: SolutionSet
    ) -> list[SpecialPoint]:
        """
        Detect saddle-node bifurcations.
        
        A saddle-node occurs when two solutions collide and annihilate.
        Detected by monitoring dω/dn (change in omega per step) for sign changes.
        
        Algorithm:
        1. Compute dω/dn for consecutive solution pairs
        2. Detect sign changes in dω/dn
        3. Mark potential bifurcation points
        
        Parameters
        ----------
        solution_set : SolutionSet
            Continuation solutions.
        
        Returns
        -------
        list[SpecialPoint]
            List of detected saddle-node points.
        """
        saddle_nodes = []
        
        if len(solution_set.omega) < 3:
            return saddle_nodes
        
        # Compute domega_dn (change in omega between consecutive solutions)
        omega_array = np.array(solution_set.omega)
        domega_dn = np.diff(omega_array)
        
        # Detect sign changes in domega_dn
        for i in range(len(domega_dn) - 1):
            product = domega_dn[i] * domega_dn[i + 1]
            
            if product < 0:  # Sign change detected
                # Determine which side has the bifurcation
                # (saddle-node is where domega_dn passes through zero)
                
                # Compute amplitude at the crossing point
                amplitude = self._compute_amplitude(solution_set.fourier[i + 1])
                
                saddle_node = SpecialPoint(
                    type='saddle_node',
                    index=i + 1,
                    omega=omega_array[i + 1],
                    amplitude=amplitude,
                    refined_omega=None,
                    multipliers=None,
                    hopf_multiplier_magnitude=None,
                    hopf_frequency=None,
                    relative_period=None,
                    is_commensurate=None,
                )
                saddle_nodes.append(saddle_node)
        
        return saddle_nodes
    
    # =========================================================================
    # Hopf (Neimark-Sacker) Detection
    # =========================================================================
    
    def detect_hopf(
        self, 
        solution_set: SolutionSet,
        stability_reports: list[StabilityReport]
    ) -> list[SpecialPoint]:
        """
        Detect Hopf bifurcations.
        
        A Hopf bifurcation occurs when a pair of complex conjugate multipliers
        cross the unit circle (|μ| = 1).
        
        Algorithm:
        1. Get magnitudes of all multipliers for each solution
        2. Track when any magnitude crosses 1
        3. Interpolate to find exact crossing point
        4. Compute relative period and check commensurability
        
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
        hopf_points = []
        
        if len(stability_reports) < 2:
            return hopf_points
        
        # For each solution, check if any multiplier crosses |μ| = 1
        for i in range(len(stability_reports) - 1):
            mags_current = np.abs(stability_reports[i].multipliers)
            mags_next = np.abs(stability_reports[i + 1].multipliers)
            
            # Check each multiplier pair
            for j in range(len(mags_current)):
                mag_current = mags_current[j]
                mag_next = mags_next[j]
                
                # Detect crossing of unit circle
                if (mag_current - 1) * (mag_next - 1) < 0:
                    # Interpolate to find exact crossing
                    omega_crossing = self._interpolate_crossing(
                        solution_set.omega[i],
                        solution_set.omega[i + 1],
                        mag_current - 1,
                        mag_next - 1
                    )
                    
                    # Get the multiplier at crossing (interpolated)
                    mult_current = stability_reports[i].multipliers[j]
                    mult_next = stability_reports[i + 1].multipliers[j]
                    
                    # Interpolate multiplier
                    t = (1 - mag_current) / (mag_next - mag_current) if (mag_next - mag_current) != 0 else 0.5
                    mult_crossing = mult_current + t * (mult_next - mult_current)
                    
                    # Compute frequency at Hopf
                    # Im(μ) = Im(exp(λT)) relates to oscillation frequency
                    # For μ = exp(i*theta), theta = arg(μ)
                    omega_hopf = solution_set.omega[i + 1] * np.abs(np.angle(mult_crossing)) / (2 * np.pi)
                    
                    # Compute relative period
                    relative_period = solution_set.omega[i + 1] / omega_hopf if omega_hopf > 0 else None
                    
                    # Check commensurability
                    is_commensurate, rational_approx = self._is_commensurate(relative_period)
                    
                    amplitude = self._compute_amplitude(solution_set.fourier[i + 1])
                    
                    hopf = SpecialPoint(
                        type='hopf',
                        index=i + 1,
                        omega=solution_set.omega[i + 1],
                        amplitude=amplitude,
                        refined_omega=omega_crossing,
                        multipliers=stability_reports[i + 1].multipliers.copy(),
                        hopf_multiplier_magnitude=np.abs(mult_crossing),
                        hopf_frequency=omega_hopf,
                        relative_period=relative_period,
                        is_commensurate=is_commensurate,
                    )
                    hopf_points.append(hopf)
        
        return hopf_points
    
    # =========================================================================
    # Combined Detection
    # =========================================================================
    
    def detect_all(
        self, 
        solution_set: SolutionSet,
        stability_reports: list[StabilityReport] = None,
        freq_domain_ode = None
    ) -> list[SpecialPoint]:
        """
        Detect all bifurcation types.
        
        Parameters
        ----------
        solution_set : SolutionSet
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
        # Compute stability reports if not provided
        if stability_reports is None:
            if freq_domain_ode is None:
                raise ValueError("Either stability_reports or freq_domain_ode must be provided")
            analyzer = FloquetAnalyzer(freq_domain_ode.ode)
            stability_reports = []
            for fourier in solution_set.fourier:
                if fourier.time_series is None:
                    fourier.compute_time_series()
                time_samples = fourier.adimensional_time_samples if hasattr(fourier, 'adimensional_time_samples') else np.linspace(0, 2*np.pi, len(fourier.time_series))
                report = analyzer.analyze(fourier.time_series, time_samples, solution_set.omega[len(stability_reports)])
                stability_reports.append(report)
        
        # Detect both types
        saddle_nodes = self.detect_saddle_nodes(solution_set)
        hopf_points = self.detect_hopf(solution_set, stability_reports)
        
        # Combine and sort by index
        all_bifurcations = saddle_nodes + hopf_points
        all_bifurcations.sort(key=lambda x: x.index)
        
        return all_bifurcations
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _compute_amplitude(self, fourier) -> float:
        """
        Compute the amplitude of a solution (max absolute value of time series).
        
        Parameters
        ----------
        fourier : Fourier
            The Fourier representation of the solution.
        
        Returns
        -------
        float
            Maximum absolute value of the time series.
        """
        if fourier.time_series is None:
            fourier.compute_time_series()
        return float(np.max(np.abs(fourier.time_series)))
    
    def _interpolate_crossing(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float
    ) -> float:
        """
        Linear interpolation to find where y = 0.
        
        Parameters
        ----------
        x0, x1 : float
            Known x values.
        y0, y1 : float
            Function values at x0, x1. Assumes y0*y1 < 0.
        
        Returns
        -------
        float
            Interpolated x where y = 0.
        """
        if y1 == y0:
            return (x0 + x1) / 2
        t = -y0 / (y1 - y0)
        return x0 + t * (x1 - x0)
    
    def _is_commensurate(
        self,
        ratio: float,
        max_denominator: int = 10
    ) -> tuple[bool, Optional[Fraction]]:
        """
        Check if a ratio is approximately a simple rational number.
        
        Parameters
        ----------
        ratio : float
            The ratio to check.
        max_denominator : int, optional
            Maximum denominator to check. Default is 10.
        
        Returns
        -------
        tuple[bool, Optional[Fraction]]
            (is_commensurate, rational_approximation)
        """
        if ratio is None or ratio <= 0:
            return False, None
        
        for p in range(1, max_denominator + 1):
            for q in range(1, max_denominator + 1):
                rational = p / q
                if abs(ratio - rational) < self.commensurability_tolerance * rational:
                    return True, Fraction(p, q)
        return False, None
```

---

## Integration with SolutionSet

Update `core.py` to add a convenience method:

```python
class SolutionSet(object):
    # ... existing methods ...
    
    def detect_bifurcations(
        self,
        stability_reports: list[StabilityReport] = None,
        freq_domain_ode = None
    ) -> list[SpecialPoint]:
        """
        Detect bifurcations in the solution set.
        
        Parameters
        ----------
        stability_reports : list[StabilityReport], optional
            Pre-computed stability reports. If None, computes on-demand.
        freq_domain_ode : FrequencyDomainFirstOrderODE, optional
            Required if stability_reports is None.
        
        Returns
        -------
        list[SpecialPoint]
            List of detected bifurcation points.
        """
        from .stability.bifurcation_detection import BifurcationDetector
        detector = BifurcationDetector()
        return detector.detect_all(self, stability_reports, freq_domain_ode)
```

---

## Exports

Update `src/pyhbm/stability/__init__.py`:

```python
from .stability_analysis import (
    FloquetAnalyzer,
    StabilityReport,
)

from .bifurcation_detection import (
    BifurcationDetector,
    SpecialPoint,
)

__all__ = [
    "FloquetAnalyzer",
    "StabilityReport",
    "BifurcationDetector",
    "SpecialPoint",
]
```

---

## Algorithm Details

### Saddle-Node Detection

1. **Input**: `SolutionSet` with `omega` list
2. **Compute**: `dω/dn = omega[i+1] - omega[i]` for all consecutive pairs
3. **Detect**: Sign change in `dω/dn` (product < 0)
4. **Record**: Index, interpolated omega, amplitude at crossing point
5. **Output**: List of `SpecialPoint` with `type='saddle_node'`

### Hopf Detection

1. **Input**: `SolutionSet` + `list[StabilityReport]`
2. **For each solution**: Get magnitudes of all Floquet multipliers
3. **Detect**: When any |μ| crosses 1 (from below or above)
4. **Interpolate**: Find exact crossing omega and multiplier
5. **Compute**:
   - Hopf frequency: from argument of multiplier at crossing
   - Relative period: ω_current / ω_hopf
   - Commensurability: check if ratio ≈ p/q for p,q ≤ 10
6. **Output**: List of `SpecialPoint` with `type='hopf'`

---

## Usage Example

```python
from pyhbm import HarmonicBalanceMethod, SolutionSet
from pyhbm.stability import FloquetAnalyzer, BifurcationDetector, SpecialPoint

# Run continuation
solver = HarmonicBalanceMethod(duffing, harmonics=[1, 3, 5, 7, 9])
solution_set = solver.solve_and_continue(...)

# Analyze stability
stability_reports = solution_set.analyze_stability(solver.freq_domain_ode)

# Detect bifurcations
detector = BifurcationDetector()
bifurcations = detector.detect_all(solution_set, stability_reports)

# Print results
for bif in bifurcations:
    print(bif)

# Or use convenience method
bifurcations = solution_set.detect_bifurcations(solver.freq_domain_ode)

# Filter by type
saddle_nodes = [b for b in bifurcations if b.type == 'saddle_node']
hopf_points = [b for b in bifurcations if b.type == 'hopf']

# Check commensurability for Hopf
for hopf in hopf_points:
    if hopf.is_commensurate:
        print(f"Commensurate Hopf at ω={hopf.omega:.4f}")
    else:
        print(f"Incommensurate Hopf at ω={hopf.omega:.4f}")
```

---

## Implementation Order

1. **Step 1**: Create `src/pyhbm/stability/bifurcation_detection.py`
   - Define `SpecialPoint` dataclass
   - Define `BifurcationDetector` class
   - Implement `_compute_amplitude`, `_interpolate_crossing`, `_is_commensurate`
   - Implement `detect_saddle_nodes`
   - Implement `detect_hopf`
   - Implement `detect_all`

2. **Step 2**: Update `src/pyhbm/stability/__init__.py`
   - Add exports for `BifurcationDetector`, `SpecialPoint`

3. **Step 3**: Update `src/pyhbm/core.py`
   - Add `detect_bifurcations` method to `SolutionSet`

4. **Step 4**: Add tests

---

## Dependencies

- `numpy` - For array operations
- `fractions` - Standard library for rational approximation
- No new external dependencies required

---

## Backward Compatibility

- All new features are additive
- No changes to existing API signatures
- `BifurcationDetector` is entirely new
- `SpecialPoint` is a new dataclass

---

## Notes

1. The implementation uses Floquet multipliers (already computed) rather than computing exponents separately
2. Hopf detection checks for |μ| = 1 crossing, not Re(λ) = 0
3. Commensurability check uses p,q ≤ 10 (can be extended if needed)
4. Amplitude is computed from max(|time_series|) - could be extended to compute specific DOF amplitude

---

## Feature 3b: Bifurcation Plotter

### Overview

Add bifurcation point markers to the `plot_FRF` function to visualize detected saddle-node and Hopf bifurcation points on the frequency response curve.

### Current Issue

The amplitude computation differs between:
1. **FRF plot**: Uses `norm(coefficients) * 2 / N` (or specific harmonic)
2. **Bifurcation detector**: Uses `max(abs(time_series))`

These don't match, so markers appear at wrong y-positions.

### Solution

1. **BifurcationDetector** should only flag indices and bifurcation types
2. **plot_FRF** should use the stored index to get y-value from user's computed `harmonic_amplitude` array

### Implementation Steps

#### Step 1: Update `src/pyhbm/stability/bifurcation_detection.py`

**A. Simplify `SpecialPoint`:**
- Remove `amplitude` field (not needed - plotter computes it from user's data)
- Keep all other fields:
  - `type`: str
  - `index`: int (KEY - used by plotter)
  - `omega`: float
  - `refined_omega`: Optional[float]
  - `multipliers`: Optional[NDArray] (interesting data)
  - `hopf_multiplier_magnitude`: Optional[float] (interesting data)
  - `hopf_frequency`: Optional[float] (interesting data)
  - `relative_period`: Optional[float]
  - `is_commensurate`: Optional[bool]

**B. Add print methods to `SpecialPoint`:**

```python
def __str__(self) -> str:
    """Short string representation."""
    base = f"{self.type} at omega={self.omega:.4f}, index={self.index}"
    if self.type == 'hopf' and self.is_commensurate is not None:
        base += f", commensurate={self.is_commensurate}"
        if self.relative_period is not None:
            base += f", period_ratio={self.relative_period:.2f}"
    return base

def print_details(self) -> str:
    """Detailed string with all bifurcation data."""
    lines = [f"Bifurcation: {self.type}"]
    lines.append(f"  Index: {self.index}")
    lines.append(f"  Omega: {self.omega:.6f}")
    if self.refined_omega is not None:
        lines.append(f"  Refined omega: {self.refined_omega:.6f}")
    
    if self.type == 'hopf':
        lines.append(f"  Is commensurate: {self.is_commensurate}")
        if self.relative_period is not None:
            lines.append(f"  Relative period: {self.relative_period:.6f}")
        if self.hopf_frequency is not None:
            lines.append(f"  Hopf frequency: {self.hopf_frequency:.6f}")
        if self.hopf_multiplier_magnitude is not None:
            lines.append(f"  Multiplier magnitude: {self.hopf_multiplier_magnitude:.6f}")
        if self.multipliers is not None:
            lines.append(f"  Multipliers:")
            for i, m in enumerate(self.multipliers):
                lines.append(f"    [{i}] {m.real:.6f} + {m.imag:.6f}j")
    
    return "\n".join(lines)
```

**C. Update detection methods:**
- `detect_saddle_nodes()`: Remove amplitude computation, only record index and omega
- `detect_hopf()`: Keep all Hopf-specific data (multipliers, frequencies, etc.)

#### Step 2: Update `src/pyhbm/io/plotting.py`

**Update `_plot_bifurcation_points` to use indices:**

```python
def _plot_bifurcation_points(bifurcations, omega_array, amplitude_array, reference_omega=None, **kwargs):
    """
    Plot bifurcation points on the FRF curve.
    
    Uses the index stored in each bifurcation point to get the correct y-value
    from the user's computed amplitude array.
    """
    marker_size = kwargs.pop('markersize', 100)
    marker_edge_width = kwargs.pop('markeredgewidth', 1.5)
    
    saddle_nodes = [b for b in bifurcations if b.type == 'saddle_node']
    hopf_points = [b for b in bifurcations if b.type == 'hopf']
    
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
                   marker='D', c='green', 
                   s=marker_size, linewidths=marker_edge_width,
                   label='Hopf', zorder=10)
    
    if len(sn_omegas) > 0 or len(hf_omegas) > 0:
        plt.legend(loc='best')
```

#### Step 3: Update `plot_FRF` function signature

Add `bifurcations` parameter:

```python
def plot_FRF(solution_set, degrees_of_freedom, harmonic=None, reference_omega=None, 
             yscale='linear', xscale='linear', show=True, stability=False, 
             time_domain_ode=None, bifurcations=None, **kwargs):
```

After plotting the FRF curve, add:

```python
# Plot bifurcation points
if bifurcations is not None:
    _plot_bifurcation_points(
        bifurcations, 
        solution_set.omega, 
        harmonic_amplitude,
        reference_omega,
        **kwargs
    )
```

### Marker Legend

| Bifurcation Type | Marker | Color | Description |
|-----------------|--------|-------|-------------|
| Saddle-Node | Circle (o) | Blue (edge only) | Fold bifurcation where solutions collide |
| Hopf | Diamond (D) | Green (filled) | Neimark-Sacker bifurcation, quasi-periodic emerges |

### Usage Example

```python
from pyhbm import plot_FRF
from pyhbm.stability import BifurcationDetector

# Run continuation
solver = HarmonicBalanceMethod(duffing, harmonics=[1, 3, 5, 7, 9])
solution_set = solver.solve_and_continue(...)

# Compute stability and detect bifurcations
analyzer = FloquetAnalyzer(duffing)
stability_reports = solution_set.analyze_stability(solver.freq_domain_ode)

detector = BifurcationDetector()
bifurcations = detector.detect_all(solution_set, stability_reports)

# Print details
for bif in bifurcations:
    print(bif)                    # Short: "hopf at omega=1.2345, index=10"
    print(bif.print_details())    # Detailed with multipliers, frequencies, etc.

# Plot with markers at correct y-positions
plot_FRF(
    solution_set, 
    degrees_of_freedom=0,
    stability=True,
    time_domain_ode=duffing,
    bifurcations=bifurcations,
    xscale='log', 
    yscale='log'
)
```

---

## Feature 3c: Save Bifurcations

### Overview

Add option to save bifurcation data along with solution set data. When `bifurcations=True` is passed to `save_solution_set`, it will compute stability reports, detect bifurcations, and save the data.

### Implementation Steps

#### Step 1: Update `src/pyhbm/io/save.py`

**A. Add helper function to convert bifurcations to serializable dict:**

```python
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
            bif_data['is_commensurate'] = bif.is_commensurate
            if bif.relative_period is not None:
                bif_data['relative_period'] = float(bif.relative_period)
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
```

**B. Modify `save_solution_set` function signature:**

Add parameters:
- `bifurcations`: bool = False (NEW - defaults to False to save computation)
- `freq_domain_ode`: required if bifurcations=True

```python
def save_solution_set(solution_set, path, 
                      harmonic_amplitude=True, 
                      amplitude=True, 
                      angular_frequency=True,
                      fourier_coefficients=False, 
                      time_series=False, 
                      adimensional_time_samples=False,
                      iterations=False, 
                      step_length=False,
                      bifurcations=False,  # NEW PARAMETER (default False)
                      MATLAB_compatible=False,
                      freq_domain_ode=None):  # REQUIRED if bifurcations=True
```

**C. Add bifurcation detection logic:**

At the end of the function (before saving), add:

```python
# Handle bifurcation data
if bifurcations:
    if freq_domain_ode is None:
        raise ValueError("freq_domain_ode is required when bifurcations=True")
    
    # Import here to avoid circular imports
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
```

#### Step 2: No changes needed to `src/pyhbm/io/__init__.py`

The `save_solution_set` function is already exported.

### Usage Examples

**1. Save without bifurcations (default, faster):**

```python
# Saves only solution data (no stability computation)
solution_set.save("data.pkl")
```

**2. Save with bifurcations (computes stability):**

```python
# Saves solution + bifurcation data
solution_set.save(
    "data.pkl", 
    bifurcations=True, 
    freq_domain_ode=solver.freq_domain_ode
)
```

**3. Load and access bifurcations:**

```python
import pickle

with open("data.pkl", 'rb') as f:
    data = pickle.load(f)

# Access bifurcation data
bifurcations_data = data.get('bifurcations', [])
for bif in bifurcations_data:
    print(f"Type: {bif['type']}, Omega: {bif['omega']}, Index: {bif['index']}")
    if bif['type'] == 'hopf':
        print(f"  Commensurate: {bif.get('is_commensurate')}")
        print(f"  Multipliers: {bif.get('multipliers')}")
```

---

## Implementation Order

1. **Step 1**: Create `src/pyhbm/stability/bifurcation_detection.py` - DONE
2. **Step 2**: Update `src/pyhbm/stability/__init__.py` - DONE
3. **Step 3**: Update `src/pyhbm/core.py` - DONE
4. **Step 4**: Update `src/pyhbm/stability/bifurcation_detection.py`
   - Remove `amplitude` field from SpecialPoint
   - Add `print_details()` method
   - Update detection methods to not compute amplitude

5. **Step 5**: Update `src/pyhbm/io/plotting.py`
   - Add `bifurcations` parameter to `plot_FRF`
   - Update `_plot_bifurcation_points` to use indices

6. **Step 6**: Update `src/pyhbm/io/save.py`
   - Add `_save_bifurcation_data()` helper
   - Add `bifurcations=False` parameter to `save_solution_set`
   - Add bifurcation computation when `bifurcations=True`

7. **Step 7**: Update example `examples/duffing_forced_nonautonomous/main.py`
   - Add bifurcation detection and plotting

8. **Step 8**: Run and validate

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/pyhbm/stability/bifurcation_detection.py` | Remove amplitude, add print_details() |
| `src/pyhbm/io/plotting.py` | Add bifurcations parameter, fix y-axis matching |
| `src/pyhbm/io/save.py` | Add bifurcations save option |
| `examples/duffing_forced_nonautonomous/main.py` | Add bifurcation detection |

---

## Backward Compatibility

- All new features are additive
- `bifurcations` parameter defaults to `None` in `plot_FRF`
- `bifurcations` parameter defaults to `False` in `save_solution_set`
- Existing calls continue to work unchanged

---

## Fixes for Bifurcation Detection Issues

### Issues Found

#### 1. Debug Print Statements (Lines 145-147)
Should be removed:
```python
if mags_current.ndim != 1 or mags_next.ndim != 1:
    print("mags_current =", mags_current)
    print("mags_next =", mags_next)
```

#### 2. Remove `is_commensurate` Field
- Remove from dataclass definition (line 21)
- Remove from `__str__` method

#### 3. Simplify `__str__` Method (Line 26)
```python
# Change from:
if self.type == 'Hopf' and self.is_commensurate is not None:
# To:
if self.type == 'Hopf':
```

#### 4. Add Missing Field to Saddle-Node Creation
Add `rational_approx_relative_period=None`:
```python
saddle_node = SpecialPoint(
    type='Saddle-Node',
    index=i + 1,
    omega_crossing=omega_array[i + 1],
    multipliers=None,
    hopf_frequency=None,
    relative_period=None,
    rational_approx_relative_period=None,  # ADD THIS
)
```

#### 5. Add Missing Field to Hopf Creation
Add `rational_approx_relative_period`:
```python
hopf = SpecialPoint(
    type='Hopf',
    index=i + 1,
    omega_crossing=omega_crossing,
    multipliers=mult_crossing,
    hopf_frequency=omega_crossing / relative_period,
    relative_period=relative_period,
    rational_approx_relative_period=self.rational_approx(relative_period),  # ADD THIS
)
```

#### 6. Implement `rational_approx` Method (Lines 242-254)
Currently returns `None` - needs implementation:
```python
def rational_approx(
    self,
    ratio: float,
    max_numerator: int = 20,
    tolerance: float = 1e-5
) -> Optional[Fraction]:
    if ratio is None:
        return None
    
    for denominator in range(1, max_numerator + 1):
        numerator = round(ratio * denominator)
        if numerator == 0:
            continue
        approx = numerator / denominator
        if abs(approx - ratio) < tolerance * ratio:
            return Fraction(numerator, denominator)
    
    return None
```

---

## Implementation Steps

### Step 1: Remove Debug Prints
Delete lines 145-147 in `src/pyhbm/stability/bifurcation_detection.py`

### Step 2: Remove `is_commensurate` Field
- Line 21: Remove `is_commensurate: Optional[bool] = None` from dataclass
- Line 26-29: Remove `is_commensurate` from `__str__` method

### Step 3: Simplify `__str__` Method
Change line 26 from:
```python
if self.type == 'Hopf' and self.is_commensurate is not None:
```
to:
```python
if self.type == 'Hopf':
```

### Step 4: Add Missing Field to Saddle-Node Creation
Around line 100-108, add `rational_approx_relative_period=None` to the SpecialPoint creation

### Step 5: Add Missing Field to Hopf Creation
Around line 174-182, add:
```python
rational_approx_relative_period=self.rational_approx(relative_period),
```
to the SpecialPoint creation

### Step 6: Implement `rational_approx` Method
Replace the incomplete method (lines 242-254) with the full implementation

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/pyhbm/stability/bifurcation_detection.py` | All fixes above |

---

## Verification

After implementation, run:
```bash
python3 examples/duffing_forced_nonautonomous/main.py
```

Expected output should show:
- No debug print statements
- Saddle-Node and Hopf bifurcations detected
- Rational approximations displayed for Hopf points
