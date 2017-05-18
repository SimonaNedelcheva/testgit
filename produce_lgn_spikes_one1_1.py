"""
This function allow us to visualize the result firing rate from a lgn cell that we can choose and then 
visualize the spikes produce by it 
"""

import numpy as np
import cPickle
from misc_functions import create_standar_indexes, create_extra_indexes
from stimuli_functions import sine_grating, ternary_noise 
from moving_dot_function import moving_dot
from kernel_functions import create_kernel 
from analysis_functions import produce_spikes, convolution
import matplotlib.pyplot as plt 
from plot_functions import visualize_firing_rate

## Time parameters

# Time scales   
dt = 0.25  # milliseconds
dt = 1
dt_kernel = 1.0  # ms
dt_stimuli = 1.0  # ms
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dt_kernel)

# Simulation time duration
T_simulation = 1 * 10 ** 3.0  # ms
remove_start = int(kernel_size * dt_kernel)
T_simulation += remove_start  # Add the size of the kernel
Nt_simulation = int(T_simulation / dt)  # Number of simulation points 
N_stimuli = int(T_simulation / dt_stimuli)  # Number of stimuli points 

## Space parameters
# visual space resolution and size  
dx = 0.05
dy = 0.05
lx = 6.0  # In degrees
ly = 6.0  # In degrees

# center-surround parameters 
factor = 1  # Controls the overall size of the center-surround pattern
sigma_center = 0.25 * factor  # Corresponds to 15'
sigma_surround = 0.5 * factor  # Corresponds to 1 degree


# sine grating spatial parameters 
K = 0.8  # Cycles per degree 
Phi = 0 * np.pi
Theta = 0 * np.pi
max_contrast = 2.4 * 2
contrast = 0.5  # Percentage
A = contrast * max_contrast 
# Temporal frequency of sine grating 
w = 3  # Hz

# Set the random set for reproducibility
seed = 31255433
np.random.seed(seed)

# Create indexes 
signal_indexes, delay_indexes, stimuli_indexes = create_standar_indexes(dt, dt_kernel,
                                                                        dt_stimuli, kernel_size, Nt_simulation)
working_indexes, kernel_times = create_extra_indexes(kernel_size, Nt_simulation)

# Create the stimuli 
#stimuli = sine_grating(dx, lx, dy, ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)
stimuli = moving_dot(lx, ly, dx, dy, Nt_simulation, dt_stimuli)

# Chose the particular cell
Nx = 10
Ny = 10
xc = np.arange(-lx/2+dx, lx/2-dx, (lx-2*dx)/Nx)
yc = np.arange(-ly/2+dy, ly/2-dy, (ly-2*dy)/Ny)
print len(xc), len(yc)
i=np.arange(0, len(xc), 1).astype(int)
j=np.arange(0, len(yc), 1).astype(int)
n=0
# Initialize the signal array to be filled 
firing_rate_ON = np.zeros((Nt_simulation, len(i)*len(j)))
firing_rate_OFF = np.zeros((Nt_simulation, len(i)*len(j)))

for xi in i:
	for yj in j:
		print 'kernel at', xi, yj
		# Create the kernel 
		kernelOFF = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center, 
			dt_kernel, kernel_size, inverse=-1, x_tra=xc[xi], y_tra=yc[yj])
		kernelON = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center, 
			dt_kernel, kernel_size, inverse=1, x_tra=xc[xi]+dx, y_tra=yc[yj]+dy)

		# Calculate the firing rate 
		for index in signal_indexes:
		    firing_rate_ON[index,n] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, 				kernelON, stimuli)
		    firing_rate_OFF[index,n] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, 			kernelOFF, stimuli)
		
		firing_rate_ON[:, n] += 10 # Add background noise 
		# Rectify the firing rate
		firing_rate_ON[firing_rate_ON[:, n] < 0, n] = 0
		firing_rate_OFF[:, n] += 10 # Add background noise 
		# Rectify the firing rate
		firing_rate_OFF[firing_rate_OFF[:, n] < 0, n] = 0
		n+=1

remove_start = int(kernel_size *dt_kernel)

#plt.figure(1)
#visualize_firing_rate(firing_rate[signal_indexes], dt, T_simulation - remove_start, label='Firing rate')
#visualize_firing_rate(firing_rate / np.max(firing_rate), dt, T_simulation, label='Firing rate')

# Produce spikes with the signal
#spike_times_thin = produce_spikes(firing_rate, dt, T_simulation, remove_start)
#spike_times_thin -= remove_start 

#y = np.ones_like(spike_times_thin) * np.max(firing_rate)

# Scale firing rate
aON=np.min(firing_rate_ON[signal_indexes, :])
bON=np.max(firing_rate_ON[signal_indexes, :])
scaled_firing_rate_ON=np.zeros((Nt_simulation-remove_start, len(i)*len(j)))
scaled_firing_rate_ON=firing_rate_ON[signal_indexes,:]*100.0/(bON-aON)-100*aON/(bON-aON)

aOFF=np.min(firing_rate_OFF[signal_indexes, :])
bOFF=np.max(firing_rate_OFF[signal_indexes, :])
scaled_firing_rate_OFF=np.zeros((Nt_simulation-remove_start, len(i)*len(j)))
scaled_firing_rate_OFF=firing_rate_OFF[signal_indexes,:]*100.0/(bOFF-aOFF)-100*aOFF/(bOFF-aOFF)

print 'minON=', aON, 'maxON=', bON
print 'minOFF=', aOFF, 'maxOFF=', bOFF



plt.figure(1)
for i in np.arange(0,Nx*Ny,1):
	plt.subplot(Nx,Ny,i+1)
	plt.plot(scaled_firing_rate_ON[:,i])
	plt.ylim([0.0, 100.0])

plt.figure(2)
for i in np.arange(0,Nx*Ny,1):
	plt.subplot(Nx,Ny,i+1)
	plt.plot(scaled_firing_rate_OFF[:,i])
	plt.ylim([0.0, 100.0])


#plt.figure(2)
#plt.imshow(stimuli[0,...],extent=[-lx/2,lx/2,ly/2,-ly/2])

#plt.figure(3)
#plt.imshow(stimuli[len(stimuli)-1, ...], extent=[-lx/2, lx/2, ly/2, -ly/2])

#plt.figure(4)
#plt.plot(firing_rate[remove_start:])
#plt.hold('on')
#plt.figure(5)
#plt.plot(spike_times_thin, y, '*', label='spikes')
#plt.legend()
#plt.ylim([9.5,9.8])
plt.show()

f=open('firing_rates_ON.cpickle',"wb")
cPickle.dump(scaled_firing_rate_ON, f, protocol=2)
f.close()

f=open('firing_rates_OFF.cpickle',"wb")
cPickle.dump(scaled_firing_rate_OFF, f, protocol=2)
f.close()
