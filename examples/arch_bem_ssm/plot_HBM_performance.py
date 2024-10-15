import os
import numpy as np
import matplotlib.pyplot as plt

# Path to the summary table file
results_dir = "./examples/arch_bem_ssm/timed_results"
summary_file_path = results_dir + "/summary_table.txt"

# Load data from the summary table
data = []
with open(summary_file_path, "r") as f:
    # Skip the header
    next(f)
    # Read the rest of the file and parse each line
    for line in f:
        harmonics, tolerance, max_step_length, goal_iterations, total_time = line.strip().split("\t")
        harmonics = len(eval(harmonics))  # Convert harmonics from string to length of array
        tolerance = float(tolerance)
        max_step_length = float(max_step_length)
        goal_iterations = int(goal_iterations)
        total_time = float(total_time)
        data.append([harmonics, tolerance, max_step_length, goal_iterations, total_time])

# Convert the data to a NumPy array for easier manipulation
data = np.array(data)

# Extract individual columns for clarity
harmonics_col = data[:, 0].astype(int)
tolerances_col = data[:, 1]
step_sizes_col = data[:, 2]
goal_iterations_col = data[:, 3].astype(int)
times_col = data[:, 4]

# Fixing a particular tolerance
fixed_tolerance = 1e-6

colors = ['k', 'r', 'b', 'g']

# 2. Plot: Time vs Harmonics for various goal iterations (fixing step size)
plt.figure(figsize=(4, 4))

# Apply a mask to filter based on goal iterations and fixed step size
y = []
for goal_iter in np.unique(goal_iterations_col):
    mask = (goal_iterations_col == goal_iter) & (step_sizes_col == 0.1) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(times_col[mask])
plt.plot(np.unique(harmonics_col), np.min(np.array(y),axis=0), label=f"Max. Step Length = 0.1", color='r', ls='--')

y = []
for goal_iter in np.unique(goal_iterations_col):
    mask = (goal_iterations_col == goal_iter) & (step_sizes_col == 0.5) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(times_col[mask])
plt.plot(np.unique(harmonics_col), np.min(np.array(y),axis=0), label=f"Max. Step Length = 0.5", color='k')

y = []
for goal_iter in np.unique(goal_iterations_col):
    mask = (goal_iterations_col == goal_iter) & (step_sizes_col == 1.0) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(times_col[mask])
plt.plot(np.unique(harmonics_col), np.min(np.array(y),axis=0), label=f"Max. Step Length = 1.0", color='b', ls=':')

#plt.title(f"Solving Time vs Harmonics for Various Goal Iterations (Step Size = {fixed_step_size})")
plt.xlabel("Number of Harmonics")
plt.ylabel("Total Solving Time (seconds)")
plt.xticks([2,3,4,5,7,9])
#plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Fixing a particular step size for comparison
fixed_step_size = 1.0

colors = ['k', 'r', 'b', 'g']

# 2. Plot: Time vs Harmonics for various goal iterations (fixing step size)
plt.figure(figsize=(4, 4))

# Apply a mask to filter based on goal iterations and fixed step size
y = []
for goal_iter in np.unique(goal_iterations_col):
    mask = (goal_iterations_col == goal_iter) & (step_sizes_col == fixed_step_size) & (tolerances_col == 1e-3)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(times_col[mask])
plt.plot(np.unique(harmonics_col), np.min(np.array(y),axis=0), label=f"Rel. Tolerance = 1e-3", color='r', ls='--')

y = []
for goal_iter in np.unique(goal_iterations_col):
    mask = (goal_iterations_col == goal_iter) & (step_sizes_col == fixed_step_size) & (tolerances_col == 1e-6)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(times_col[mask])
plt.plot(np.unique(harmonics_col), np.min(np.array(y),axis=0), label=f"Rel. Tolerance = 1e-6", color='k')

y = []
for goal_iter in np.unique(goal_iterations_col):
    mask = (goal_iterations_col == goal_iter) & (step_sizes_col == fixed_step_size) & (tolerances_col == 1e-9)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(times_col[mask])
plt.plot(np.unique(harmonics_col), np.min(np.array(y),axis=0), label=f"Rel. Tolerance = 1e-9", color='b', ls=':')

#plt.title(f"Solving Time vs Harmonics for Various Goal Iterations (Step Size = {fixed_step_size})")
plt.xlabel("Number of Harmonics")
plt.ylabel("Total Solving Time (seconds)")
plt.xticks([2,3,4,5,7,9])
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
