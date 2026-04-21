# Tracking Bifurcations Plan: Hopf Winding Number Correction

## Problem Statement

In `src/pyhbm/stability/bifurcation_detection.py`, the Hopf bifurcation detection uses `np.abs(np.angle(mult_crossing[j]))` which returns values in [0, 2π]. This loses information about how many times the angle has wound around the unit circle.

**Examples:**
- `angle = 0` → fixed point, frequency = 0
- `angle = 2π` → same frequency as original limit cycle
- `angle = 4π` → frequency doubles, period halves

## Root Cause

The `detect_hopf` method at lines 113-182 computes the angle at the crossing point using only the raw `np.angle()` result, which is always wrapped to [-π, π] (or [0, 2π] with abs).

## Solution Overview

Track cumulative (unwrapped) angles across the continuation sequence to preserve winding number information. When the angle "wraps around" (crosses ±π), add or subtract 2π to maintain continuity.

---

## Implementation Steps

### Step 1: Add Helper Method for Computing Unwrapped Angles

**Location:** `BifurcationDetector` class, after `__init__` (around line 67)

**Method signature:**
```python
def _compute_unwrapped_angles(
    self,
    stability_reports: list[StabilityReport],
    multiplier_index: int
) -> NDArray[np.floating]:
    """
    Compute cumulative unwrapped angles for a specific multiplier across all solutions.

    Parameters
    ----------
    stability_reports : list[StabilityReport]
        Pre-computed stability reports for each solution.
    multiplier_index : int
        Index of the multiplier to track.

    Returns
    -------
    NDArray[np.floating]
        Cumulative unwrapped angles (in radians) for each solution point.
    """
```

**Implementation logic:**
1. Extract raw angles for the given multiplier index from all stability reports
   ```python
   raw_angles = np.array([np.angle(report.multipliers[multiplier_index]) 
                          for report in stability_reports])
   ```

2. Initialize unwrapped array with first angle
   ```python
   unwrapped = np.zeros_like(raw_angles)
   unwrapped[0] = raw_angles[0]
   ```

3. Iterate through consecutive points and detect wrap-arounds
   ```python
   for i in range(1, len(raw_angles)):
       delta = raw_angles[i] - raw_angles[i-1]
       # Detect wrap-around: if jump > π, it wrapped forward (subtract 2π)
       # if jump < -π, it wrapped backward (add 2π)
       if delta > np.pi:
           delta -= 2 * np.pi
       elif delta < -np.pi:
           delta += 2 * np.pi
       unwrapped[i] = unwrapped[i-1] + delta
   ```

4. Return unwrapped angles array

---

### Step 2: Add Storage for Unwrapped Angles in `detect_all`

**Location:** `detect_all` method (lines 184-225)

**Add after stability reports are computed (around line 217):**
```python
# Pre-compute unwrapped angles for each multiplier pair
# We only need to track complex conjugate pairs (2 at a time)
num_multipliers = len(stability_reports[0].multipliers)
unwrapped_angles = {}

for mult_idx in range(0, num_multipliers, 2):
    unwrapped_angles[mult_idx] = self._compute_unwrapped_angles(
        stability_reports, mult_idx
    )
```

---

### Step 3: Modify `detect_hopf` Method Signature

**Location:** Line 113-117

**Change from:**
```python
def detect_hopf(
    self,
    solution_set,
    stability_reports: list[StabilityReport]
) -> list['SpecialPoint']:
```

**To:**
```python
def detect_hopf(
    self,
    solution_set,
    stability_reports: list[StabilityReport],
    unwrapped_angles: Optional[dict[int, NDArray[np.floating]]] = None
) -> list['SpecialPoint']:
```

**Add import for Optional if not already present** (line 2 already has it)

---

### Step 4: Update Angle Computation at Crossing

**Location:** Line 165 in `detect_hopf`

**Current code:**
```python
angle_hopf = np.abs(np.angle(mult_crossing[j]))
```

**Replace with interpolation of unwrapped angle:**
```python
# Use cumulative unwrapped angle if available, otherwise fallback to wrapped
if unwrapped_angles is not None and j in unwrapped_angles:
    # Interpolate the unwrapped angle at the crossing point
    cumulative_angles = unwrapped_angles[j]
    angle_hopf = cumulative_angles[i] + t_crossing * (cumulative_angles[i + 1] - cumulative_angles[i])
else:
    # Fallback to original wrapped angle behavior
    angle_hopf = np.angle(mult_crossing[j])
```

**Note:** We preserve the sign of the angle (negative angles have meaning for the direction of rotation). No `np.abs()` is used. conjugates.

---

### Step 5: Update `detect_all` to Pass Unwrapped Angles

**Location:** Line 220 in `detect_all`

**Current code:**
```python
hopf_points = self.detect_hopf(solution_set, stability_reports)
```

**Replace with:**
```python
hopf_points = self.detect_hopf(solution_set, stability_reports, unwrapped_angles)
```

---

### Step 6: Verify Period Calculation Logic

**Location:** Lines 167-178

**Current logic:**
```python
relative_frequency = angle_hopf / (2 * np.pi)  # less than 1

relative_period = 1.0 / relative_frequency if relative_frequency != 0.0 else None
```

**Verify this still works:**
- If `angle_hopf = 0` → `relative_frequency = 0` → `relative_period = None` (fixed point)
- If `angle_hopf = 2π` → `relative_frequency = 1` → `relative_period = 1.0` (same period)
- If `angle_hopf = 4π` → `relative_frequency = 2` → `relative_period = 0.5` (period halves)
- If `angle_hopf = π` → `relative_frequency = 0.5` → `relative_period = 2.0` (period doubles)

This logic is correct for the unwrapped angles.

---

### Step 7: Add Unit Tests

**Location:** New test file or existing test file for bifurcation detection

**Test cases to add:**

1. **Test unwrapped angle computation**
   - Create mock stability reports with angles that wrap around
   - Verify cumulative angles are computed correctly

2. **Test Hopf detection with winding**
   - Simulate a period-doubling scenario (angle goes 0 → 2π → 4π)
   - Verify relative_period is computed correctly

3. **Test edge case: angle near ±π**
   - Ensure wrap-around detection works when angle crosses the boundary

---

## Summary of Changes

| File | Line(s) | Change |
|------|---------|--------|
| `bifurcation_detection.py` | After line 67 | Add `_compute_unwrapped_angles` helper method |
| `bifurcation_detection.py` | 113-117 | Add `unwrapped_angles` parameter to `detect_hopf` |
| `bifurcation_detection.py` | ~165 | Use unwrapped angle interpolation at crossing |
| `bifurcation_detection.py` | ~217-223 | Compute unwrapped angles in `detect_all` |
| `bifurcation_detection.py` | ~220 | Pass unwrapped angles to `detect_hopf` |

---

## Notes

1. **Complex conjugate pairs:** The code tracks multiplier pairs at indices (0,1), (2,3), etc. We only need to track the even index since both angles in a pair will have the same winding behavior.

2. **Fallback behavior:** If `unwrapped_angles` is not provided (for backward compatibility or when called directly), the method falls back to the original wrapped angle behavior.

3. **Negative angles:** The sign of the cumulative angle is preserved to maintain information about the direction of rotation (clockwise vs counter-clockwise in the complex plane).
