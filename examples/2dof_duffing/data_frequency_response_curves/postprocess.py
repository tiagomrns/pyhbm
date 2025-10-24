from matplotlib import pyplot as plt
from scipy.io import loadmat

ax = plt.figure().add_subplot(projection='3d')

dof = 0
clist = [0.005, 0.0152, 0.02, 0.05, 0.1]
colors = ['k', 'g', 'r', 'c', 'y']

for c, color in zip(clist, colors):
    data = loadmat("examples/2dof_duffing/data_frequency_response_curves/frf_2dof_duffing_c=%.4f.mat" % c)
    angular_frequency = data['angular_frequency'].ravel()
    amplitude = data['amplitude']
    
    plt.plot(angular_frequency, amplitude[:, dof, 0], amplitude[:, 2, 0], label="c=%.4f" % c, color=color)
    
    
data = loadmat("examples/2dof_duffing/data_frequency_response_curves/frf_2dof_duffing_c=%.4f_isola.mat" % clist[0])
angular_frequency = data['angular_frequency'].ravel()
amplitude = data['amplitude']



plt.plot(angular_frequency, amplitude[:, dof, 0], amplitude[:, 2, 0], color=colors[0])
    
    
plt.xlabel("Angular Frequency $\omega$")
plt.ylabel("Amplitude $\max_{t \in [0, T]} |q_%d(t)|$" % dof)
ax.set_zlabel("Amplitude $\max_{t \in [0, T]} |q_2(t)|$")
plt.title("Frequency Response Curves for 2DoF Duffing System")
plt.legend()
plt.show()

