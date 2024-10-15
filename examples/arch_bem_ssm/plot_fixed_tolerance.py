import os
import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(6.5, 6.5))

#########################################
#                                       #
#               MatCont                 #
#                                       #
#########################################

# Path to the summary table file
results_dir = "./examples/arch_bem_ssm/timed_results"
summary_file_path = results_dir + "/matcont_performance.txt"

# Load data from the summary table
data = []
with open(summary_file_path, "r") as f:
    # Skip the header
    next(f)
    # Read the rest of the file and parse each line
    for line in f:
        Ncol, Ntst, Ntime, tolerance, max_step_length,	total_time = line.strip().split("\t")
        Ncol =  int(Ncol)
        Ntst = int(Ntst)
        Ntime = int(Ntime)
        tolerance = float(tolerance)
        max_step_length = float(max_step_length)
        total_time = float(total_time)
        data.append([Ncol, Ntst, Ntime, tolerance, max_step_length,	total_time])

# Convert the data to a NumPy array for easier manipulation
data = np.array(data)

# Extract individual columns for clarity
Ncol = data[:, 0].astype(int)
Ntst = data[:, 1].astype(int)
time_resolution = data[:, 2].astype(int)
tolerances_col = data[:, 3]
step_sizes_col = data[:, 4]
times_col = data[:, 5]

# Fixing a particular tolerance
fixed_tolerance = 1e-6

y = []
for N in np.unique(time_resolution):
    mask = (time_resolution == N) & (step_sizes_col == 0.1) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(min(times_col[mask].ravel()))
plt.plot(np.unique(time_resolution)[:-1], y[:-1], color='k', ls='--', marker='*')

y = []
for N in np.unique(time_resolution):
    mask = (time_resolution == N) & (step_sizes_col == 0.5) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(min(times_col[mask].ravel()))
plt.plot(np.unique(time_resolution)[:-1], y[:-1], color='k', marker='x')

y = []
for N in np.unique(time_resolution):
    mask = (time_resolution == N) & (step_sizes_col == 1.0) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(min(times_col[mask].ravel()))
plt.plot(np.unique(time_resolution)[:-1], y[:-1], color='k', ls=':', marker='v')



#########################################
#                                       #
#               PyHBM                   #
#                                       #
#########################################


# Path to the summary table file
results_dir = "./examples/arch_bem_ssm/timed_results"
summary_file_path = results_dir + "/pyhbm_performance.txt"

# Load data from the summary table
data = []
with open(summary_file_path, "r") as f:
    # Skip the header
    next(f)
    # Read the rest of the file and parse each line
    for line in f:
        harmonics, tolerance, max_step_length, goal_iterations, total_time = line.strip().split("\t")
        harmonics = max(eval(harmonics))  # Convert harmonics from string to length of array
        tolerance = float(tolerance)
        max_step_length = float(max_step_length)
        goal_iterations = int(goal_iterations)
        total_time = float(total_time)
        data.append([harmonics, tolerance, max_step_length, goal_iterations, total_time])

# Convert the data to a NumPy array for easier manipulation
data = np.array(data)

# Extract individual columns for clarity
time_resolution = data[:, 0].astype(int)*2
tolerances_col = data[:, 1]
step_sizes_col = data[:, 2]
goal_iterations_col = data[:, 3].astype(int)
times_col = data[:, 4]

# Apply a mask to filter based on goal iterations and fixed step size
y = []
for N in np.unique(time_resolution):
    mask = (time_resolution == N) & (step_sizes_col == 0.1) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(min(times_col[mask].ravel()))
plt.plot(np.unique(time_resolution)[:-1], y[:-1], color='r', ls='--', marker='*')

y = []
for N in np.unique(time_resolution):
    mask = (time_resolution == N) & (step_sizes_col == 0.5) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(min(times_col[mask].ravel()))
plt.plot(np.unique(time_resolution)[:-1], y[:-1], color='r', marker='x')

y = []
for N in np.unique(time_resolution):
    mask = (time_resolution == N) & (step_sizes_col == 1.0) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(min(times_col[mask].ravel()))
plt.plot(np.unique(time_resolution)[:-1], y[:-1], color='r', ls=':', marker='v')


#########################################
#                                       #
#              Shooting                 #
#                                       #
#########################################


# Path to the summary table file
results_dir = "./examples/arch_bem_ssm/timed_results"
summary_file_path = results_dir + "/shooting_performance.txt"

# Load data from the summary table
data = []
with open(summary_file_path, "r") as f:
    # Skip the header
    next(f)
    # Read the rest of the file and parse each line
    for line in f:
        _, Ntime, max_step_length, tol_Sh, total_time = line.strip().split("\t")
        Ntime = int(Ntime)
        tolerance = float(tol_Sh)
        max_step_length = float(max_step_length)
        total_time = float(total_time)
        data.append([Ntime, tolerance, max_step_length, total_time])

# Convert the data to a NumPy array for easier manipulation
data = np.array(data)

# Extract individual columns for clarity
time_resolution = data[:, 0].astype(int)
tolerances_col = data[:, 1]
step_sizes_col = data[:, 2]
times_col = data[:, -1]

# Apply a mask to filter based on goal iterations and fixed step size
y = []
for N in np.unique(time_resolution):
    mask = (time_resolution == N) & (step_sizes_col == 0.1) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(min(times_col[mask].ravel()))
plt.plot(np.unique(time_resolution), y, color='b', ls='--', marker='*')

y = []
for N in np.unique(time_resolution):
    mask = (time_resolution == N) & (step_sizes_col == 0.5) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(min(times_col[mask].ravel()))
plt.plot(np.unique(time_resolution), y, color='b', marker='x')

y = []
for N in np.unique(time_resolution):
    mask = (time_resolution == N) & (step_sizes_col == 1.0) & (tolerances_col == fixed_tolerance)
    if np.any(mask):  # Only plot if there is data for this combination
        y.append(min(times_col[mask].ravel()))
plt.plot(np.unique(time_resolution), y, color='b', ls=':', marker='v')


plt.xlabel(r"Time resolution, $N$")
plt.ylabel("Total Solving Time (seconds)")
#plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()