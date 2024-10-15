import os
import numpy as np
import matplotlib.pyplot as plt

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
Ntime = data[:, 2].astype(int)
tolerance = data[:, 3]
max_step_length = data[:, 4]
total_time = data[:, 5]

# Fixing a particular tolerance
fixed_tolerance = 1e-6

# 2. Plot: Time vs Harmonics for various goal iterations (fixing step size)
plt.figure(figsize=(4, 4))

# Apply a mask to filter based on goal iterations and fixed step size
mask = (max_step_length == 0.1) & (tolerance == fixed_tolerance)
if np.any(mask):  # Only plot if there is data for this combination
    plt.plot(Ntime[mask], total_time[mask], label=f"Max. Step Length = 0.1", color='r', ls='--')

mask = (max_step_length == 0.5) & (tolerance == fixed_tolerance)
if np.any(mask):  # Only plot if there is data for this combination
    plt.plot(Ntime[mask], total_time[mask], label=f"Max. Step Length = 0.5", color='k')

mask = (max_step_length == 1.0) & (tolerance == fixed_tolerance)
if np.any(mask):  # Only plot if there is data for this combination
    plt.plot(Ntime[mask], total_time[mask], label=f"Max. Step Length = 1.0", color='b', ls=':')

#plt.title(f"Solving Time vs Harmonics for Various Goal Iterations (Step Size = {fixed_step_size})")
plt.xlabel("Number of Collocation Points")
plt.ylabel("Total Solving Time (seconds)")
plt.xticks([30,45,80,120])
#plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Fixing a particular step size for comparison
fixed_step_length = 1.0

# 2. Plot: Time vs Harmonics for various goal iterations (fixing step size)
plt.figure(figsize=(4, 4))

# Apply a mask to filter based on goal iterations and fixed step size
mask = (max_step_length == fixed_step_length) & (tolerance == 1e-3)
if np.any(mask):  # Only plot if there is data for this combination
    plt.plot(Ntime[mask], total_time[mask], label=f"Rel. Tolerance = 1e-3", color='r', ls='--')

mask = (max_step_length == fixed_step_length) & (tolerance == 1e-6)
if np.any(mask):  # Only plot if there is data for this combination
    plt.plot(Ntime[mask], total_time[mask], label=f"Rel. Tolerance = 1e-6", color='k')

mask = (max_step_length == fixed_step_length) & (tolerance == 1e-9)
if np.any(mask):  # Only plot if there is data for this combination
    plt.plot(Ntime[mask], total_time[mask], label=f"Rel. Tolerance = 1e-9", color='b', ls=':')

#plt.title(f"Solving Time vs Harmonics for Various Goal Iterations (Step Size = {fixed_step_size})")
plt.xlabel("Number of Collocation Points")
plt.ylabel("Total Solving Time (seconds)")
plt.xticks([30,45,80,120])
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()