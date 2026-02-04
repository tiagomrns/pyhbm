import numpy as np
from matplotlib import pyplot as plt

xmax = 4.0
ymax = 1.3

plt.figure(figsize=(4*xmax, 4*ymax), dpi=500)

initial_conditions = 1.0*np.array([[np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2*np.pi, 100, endpoint=False)]).T

plt.plot([0,0], [-ymax, ymax], color='black', lw=1)
plt.plot([-xmax,xmax], [0, 0], color='black', lw=1)

def time_evolution(init, t):
    x0 = init[0]
    y0 = init[1]
    x = x0 / np.sqrt(1+(x0**2)*t) # x' = -x^3/2
    # x = x0 / (1-x0*t) # x' = x^2
    y = y0 * np.exp(-t)
    return np.vstack((x,y))


def plot_center_manifold(point):
    x0 = point[0]
    y0 = point[1]
    aux = np.log(np.abs(y0)*2/3) + (1/x0)**2
    print(aux)
    x_max = np.pow(np.max((np.pow(xmax,-2.0), np.log(np.abs(y0)/ymax) + (1/x0)**2)), -0.5)
    if x0 < -0.0001:
        x = np.linspace(-x_max, 0, 100)
    elif x0 > 0.0001:
        x = np.linspace(0, x_max, 100)
    else:
        return
    #y = point[1] * np.exp((1/x) - (1/x0))
    y = point[1] * np.exp((1/x0)**2 - (1/x)**2)
    plt.plot(x, y, color='gray', lw=0.5)
    
for point in initial_conditions[:,::1].T:
    plot_center_manifold(point)
    

for t in np.linspace(-1,5,6, endpoint=False):
    state = time_evolution(initial_conditions, t)
    plt.plot(state[0], state[1], marker='o', markersize=1, lw=0)

plt.axis("equal")
plt.axis([-xmax, xmax, -ymax, ymax])
plt.savefig("/Users/tiago/Desktop/AM-TUM/Teaching/Student_Theses/Joana_Pereira_Non-intrusive_DPIM_SoSe_2025/centre_manifold/Figure_1.png")