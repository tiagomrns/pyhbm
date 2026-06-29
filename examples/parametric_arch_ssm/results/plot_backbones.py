theta_list = [0.0, 0.5] #[-1.0, -0.5, 0.0, 0.5, 1.0]
z_order_list = [3, 5, 7]

import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

results_dir = Path(__file__).parent
styles = {3: ':', 5: '--', 7: '-'}
colors = {-1.0: 'r', 0.0: 'g', 1.0: 'b'}

rom = np.load(results_dir.parent / "rom.npz")
dof = 224 - 1
#dof = np.argmax(np.abs(rom["W"][:,0,0]))
print("parametric dof =", dof)
W = rom["W"][dof,0]          # complex128, shape (FOM, ORD, L), Fortran order
print("W[0] =", W[0])
R = rom["R"]          # complex128, shape (NVAR, L)
exponents = rom["exponents"].T  # int32, shape (NVAR, L)
# exponents[:, m] = [a, b, c] is the multiindex for W[:, :, m]

#for exp, coeff in zip(exponents, W):
#    print(f"exp = {exp} \t coeff = {coeff}")

print(f"\n\n")

MAX_DISP = 5

class plot_parametric_arch():
    def compute_max_displacement(xy_time, theta, order):
        x = xy_time[:,0]
        y = xy_time[:,1]
        z =  x + 1j * y
        zconj = x - 1j * y
        out = np.zeros_like(z)
        for exp, coeff in zip(exponents, W):
            # if sum(exp[:-1]) == 0: continue
            if sum(exp[:-1]) > order: break
            term = (z**exp[0]) * (zconj**exp[1]) * (theta**exp[2])
            out += coeff * term
        #print(np.max(np.imag(out)))
        return np.max(np.abs(np.real(out)), axis=0)

    def plot_displacement_backbone(theta, z_order):

        fname = f"parametric_arch_ssm_theta_{theta:.1f}_z_truncation_order_{z_order}.h5"
        with open(results_dir / fname, 'rb') as f:
            data = pickle.load(f)
            
        omega = abs(np.array(data['angular_frequency']))
            
        displacement = []
        for sol_idx, xy_time in enumerate(data["time_series"]):
            disp = plot_parametric_arch.compute_max_displacement(xy_time, theta, z_order)
            if disp > MAX_DISP: break
            displacement.append(disp)
        displacement = np.array(displacement) #* (0.675660564517682/0.6764014527465906)
            
        plt.plot((omega[:len(displacement)]-omega[0])/omega[0], displacement, ls=styles[z_order], color=colors[theta])
        
    def plot(pairs):
        for (theta, z_order) in pairs:
            plot_parametric_arch.plot_displacement_backbone(theta, z_order)  
   
   
              
class plot_reference():
    def compute_max_displacement(W_ref, exponents_ref, xy_time, arch_mm):
        x = xy_time[:,0]
        y = xy_time[:,1]
        z =  x + 1j * y
        zconj = x - 1j * y
        out = np.zeros_like(z)
        for exp, coeff in zip(exponents_ref, W_ref):
            #if sum(exp) > 1: break
            term = (z**exp[0]) * (zconj**exp[1])
            out += coeff * term
        return np.max(np.abs(np.real(out)), axis=0)

    def plot_displacement_backbone(arch_mm):

        fname = f"reference_arch_ssm_{arch_mm}mm.h5"
        with open(results_dir / fname, 'rb') as f:
            data = pickle.load(f)
            
        omega = abs(np.array(data['angular_frequency']))

        rom_ref = np.load(results_dir.parent / f"reference_arches/rom_{arch_mm}mm.npz")
        dof = 224 - 1
        #dof = np.argmax(np.abs(rom_ref["W"][:,0,0]))
        print("reference dof =", dof)
        W_ref = rom_ref["W"][dof,0]          # complex128, shape (FOM, ORD, L), Fortran order
        print("W_ref[0] =", W_ref[0])
        exponents_ref = rom_ref["exponents"].T  # int32, shape (NVAR, L)
        
        #for exp, coeff in zip(exponents_ref, W_ref):
        #    print(f"exp = {exp} \t coeff = {coeff}")
            
        displacement = []
        for sol_idx, xy_time in enumerate(data["time_series"]):
            disp = plot_reference.compute_max_displacement(W_ref, exponents_ref, xy_time, arch_mm)
            if disp > MAX_DISP: break
            displacement.append(disp)
            
        plt.plot((omega[:len(displacement)]-omega[0])/omega[0], displacement, color="k")
        
    def plot(arch_mm):
        plot_reference.plot_displacement_backbone(arch_mm)  


plt.figure(figsize=(4, 4), dpi=150)

plot_reference.plot(arch_mm=0)
plot_parametric_arch.plot([(-1.0, 3), (-1.0, 5), (-1.0, 7)])


plot_reference.plot(arch_mm=5)
plot_parametric_arch.plot([(0.0, 7)])#[(0.0, 3), (0.0, 5), (0.0, 7)])


plot_reference.plot(arch_mm=10)
plot_parametric_arch.plot([(1.0, 3), (1.0, 5), (1.0, 7)])



plt.ylim([0,5])
plt.xlim([-0.04,0.07])
plt.show()
     

