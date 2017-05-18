"""
This function allow us to visualize the result firing rate from a lgn cell that we can choose and then 
visualize the spikes produce by it 
"""
import nest
import numpy as np
from misc_functions import create_standar_indexes, create_extra_indexes
from stimuli_functions import sine_grating, ternary_noise 
from moving_dot_function import moving_dot
from kernel_functions import create_kernel 
from analysis_functions import produce_spikes, convolution
import matplotlib.pyplot as plt 
from plot_functions import visualize_firing_rate
import cPickle	
#import nest.raster_plot

## Time parameters
# Time scales   
dt = 1
dt_kernel = 1.0  # ms
dt_stimuli = 1.0  # ms
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dt_kernel)

# Simulation time duration
#T_simulation = 1 * 10 ** 3.0  # ms
#remove_start = int(kernel_size * dt_kernel)
#T_simulation += remove_start  # Add the size of the kernel
#Nt_simulation = int(T_simulation / dt)  # Number of simulation points 
#N_stimuli = int(T_simulation / dt_stimuli)  # Number of stimuli points 

## Space parameters
# visual space resolution and size  
dx = 0.05
dy = 0.05
lx = 6.0  # In degrees
ly = 6.0  # In degrees

# center-surround parameters 
factor = 1  # Controls the overall size of the center-surround pattern
sigma_center = 0.75 * factor  # Corresponds to 15'
sigma_surround = 1 * factor  # Corresponds to 1 degree

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

# Initialize the signal array to be filled 
#firing_rate = np.zeros((Nt_simulation,2))
lxk=2
lyk=2

stimuli=np.zeros((N_stimuli,int(lx/dx),int(ly/dy)))
'''
# Create the stimuli 
#stimuli = sine_grating(dx, lx, dy, ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)
stimuli = moving_dot(lx, ly, dx, dy, N_stimuli, dt_stimuli)
'''
# Chose the particular cell 
xc = 0
yc = 0

#LOAD ON,OFF
f = open("firing_rates_ON.cpickle", "rb")
scaled_firing_rateON = cPickle.load(f)
print scaled_firing_rateON

f = open("firing_rates_OFF.cpickle", "rb")
scaled_firing_rateOFF = cPickle.load(f)
print scaled_firing_rateOFF

dimensionON = len(scaled_firing_rateON[0,:])
print "dimension ON:", dimensionON

dimensionOFF = len(scaled_firing_rateOFF[0,:])
print "dimension OFF:", dimensionOFF

print len(scaled_firing_rateOFF)

dimension = dimensionON
Nx = np.sqrt(dimensionON).astype(int)
Ny = Nx

t_sim = len(scaled_firing_rateON[:,0])
t = np.arange(0, t_sim).astype(int)

v_ON = np.zeros((t_sim, dimensionON))
v_OFF = np.zeros((t_sim, dimensionOFF))

spikeON = np.zeros((t_sim, dimensionON))
spikeOFF = np.zeros((t_sim, dimensionOFF))

kernelON = np.zeros((kernel_duration,int(lxk/dx),int(lyk/dy)))
kernelOFF = np.zeros((kernel_duration,int(lxk/dx),int(lyk/dy)))

params = {"V_th" : -59.2 }
neuronON = nest.Create("iaf_chxk_2008", dimensionON)
nest.SetStatus(neuronON, params)

#params = {"V_th" : -59.2 }
neuronOFF = nest.Create("iaf_chxk_2008", dimensionOFF)
nest.SetStatus(neuronOFF, params)

#step = ((lx/dx)/3)
#stx = np.arange(0, lx/dx, step)
#sty = np.arange(0, ly/dy, step)
xn = np.arange(0,dimension,1)
nsON = np.zeros((dimensionON,1))
nsOFF = np.zeros((dimensionOFF,1))
'''
kernelON = create_kernel(dx, lxk, dy, lyk, sigma_surround, sigma_center,dt_kernel, kernel_size, inverse= 1, x_tra=xc, y_tra=yc)
kernelOFF = create_kernel(dx, lxk, dy, lyk, sigma_surround, sigma_center,dt_kernel, kernel_size, inverse= -1, x_tra=xc, y_tra=yc)
for index in signal_indexes:
	for xs,i in zip(stx,x):
		for ys,j in zip(sty,y):
	   		firing_rateON[index,i,j] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernelON, stimuli[:, xs:xs+step, ys:ys+step])         
			firing_rateOFF[index,i,j] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernelOFF, stimuli[:, xs:xs+step, ys:ys+step])
# Create the kernel
#for j in [-1, 1]:
#	kernel[count,...] = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center, dt_kernel, kernel_size, inverse= j, x_tra=xc, y_tra=yc)
	# Calculate the firing rate 
	#for index in signal_indexes:
	#  		firing_rate[index,count] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel[count,...], stimuli)

firing_rateON += 10 	# Add background noise 
firing_rateON[ firing_rateON < 0] = 0  	# Rectify the firing rate
firing_rateOFF += 100
firing_rateOFF[ firing_rateOFF < 0] = 0 

# Scale firing rate
a1 = np.min(firing_rateON[signal_indexes,...])
b1 = np.max(firing_rateON[signal_indexes,...])
scaled_firing_rateON = firing_rateON[signal_indexes,...]*100.0/(b1-a1)-100*a1/(b1-a1)
a2 = np.min(firing_rateOFF[signal_indexes,...])
b2 = np.max(firing_rateOFF[signal_indexes,...])
scaled_firing_rateOFF=firing_rateOFF[signal_indexes,...]*100.0/(b2-a2)-100*a2/(b2-a2)
'''

multimeterON = nest.Create("multimeter", dimensionON)
nest.SetStatus(multimeterON, {"withtime":True, "record_from":["V_m"]})	
dc_ON = nest.Create("dc_generator", dimensionON)
sd_ON = nest.Create("spike_detector", dimensionON)

multimeterOFF = nest.Create("multimeter", dimensionOFF)
nest.SetStatus(multimeterOFF, {"withtime":True, "record_from":["V_m"]})	
dc_OFF = nest.Create("dc_generator", dimensionOFF)
sd_OFF = nest.Create("spike_detector", dimensionOFF)

for xi in xn:
	nest.Connect([multimeterON[xi]], [neuronON[xi]])
	nest.Connect([dc_ON[xi]], [neuronON[xi]])
	nest.Connect([neuronON[xi]], [sd_ON[xi]])
	
	nest.Connect([multimeterOFF[xi]], [neuronOFF[xi]])
	nest.Connect([dc_OFF[xi]], [neuronOFF[xi]])
	nest.Connect([neuronOFF[xi]], [sd_OFF[xi]])


for i in t:
	ndc=0
	for count in np.arange(0,dimension,1):
		nest.SetStatus([dc_ON[ndc]], {"amplitude" : scaled_firing_rateON[i,count]})
		nest.SetStatus([dc_OFF[ndc]], {"amplitude" : scaled_firing_rateOFF[i,count]})
		ndc+=1

	nest.Simulate(1.0)

	for xi in xn:
		v_ON[:i,xi] = nest.GetStatus(multimeterON)[xi]["events"]["V_m"]
		v_OFF[:i,xi] = nest.GetStatus(multimeterOFF)[xi]["events"]["V_m"]

		stON = nest.GetStatus([sd_ON[xi]], "events")[0]["times"]
		stOFF = nest.GetStatus([sd_OFF[xi]], "events")[0]["times"]

		spikeON[:len(stON),xi] = stON
		spikeOFF[:len(stOFF),xi] = stOFF

		nsON[xi] = len(stON)
		nsOFF[xi] = len(stOFF)

print 'v_ON',v_ON
print 'v_OFF',v_OFF

print 'spikeON',spikeON
print 'spikeOFF',spikeOFF


'''
cnt=1
plt.figure(1)
for count in np.arange(0,120,40):
	for count1 in np.arange(0,120,40):
		plt.subplot(3,3,cnt)
		plt.imshow(stimuli[0,count:count+40,count1:count1+40],extent=[-lx/2,lx/2,ly/2,-ly/2])
		cnt+=1

plt.figure(2)
plt.subplot(2,2,1)
plt.imshow(kernelON[0,...],extent=[-lxk/2,lxk/2,lyk/2,-lyk/2])
plt.colorbar() 

plt.subplot(2,2,2)
plt.imshow(kernelOFF[0,...],extent=[-lxk/2,lxk/2,lyk/2,-lyk/2])
plt.colorbar()
'''
cnt=1
plt.figure(3)
for count in np.arange(0,dimensionON,1):	
	plt.subplot(Nx,Ny,cnt)
	plt.ylabel("firing rate ON")
	plt.plot(scaled_firing_rateON[:,count])
	plt.ylim([0,100])
	cnt+=1


cnt=1
plt.figure(4)
for count in np.arange(0,dimensionON,1):
	plt.xlabel("Time")
	plt.ylabel("spike ON")	
	y = np.ones_like(spikeON[:nsON[count],count]) 
	y*=cnt
	plt.plot(spikeON[:nsON[count],count],y, '*', label='spikes')
	plt.xlim([0,t_sim])
	cnt+=1

cnt=1
plt.figure(5)
for count in np.arange(0,dimensionON,1):
	plt.subplot(Nx,Ny,cnt)
	plt.plot(v_ON[:,count])
	plt.ylabel('Vm - ON')
	plt.ylim([-80,-50])
	cnt+=1


cnt=1
plt.figure(6)	
for count in np.arange(0,dimensionOFF,1):	
	plt.subplot(Nx,Ny,cnt)
	plt.ylabel("firing rate OFF")
	plt.plot(scaled_firing_rateOFF[:,count])
	plt.ylim([0,100])
	cnt+=1

cnt=1
plt.figure(7)
for count in np.arange(0,dimensionOFF,1): 
		plt.xlabel("Time")
		plt.ylabel("spike OFF")
		y = np.ones_like(spikeOFF[:nsOFF[count],count])
		y+=cnt
		plt.plot(spikeOFF[:nsOFF[count],count],y, '*', label='spikes')
		plt.xlim([0,t_sim])
		cnt+=1
		

cnt=1
plt.figure(8)
for count in np.arange(0,Nx*Ny,1):
	plt.subplot(Nx,Ny,cnt)
	plt.plot(v_OFF[:,count])
	plt.ylabel('Vm - OFF')
	plt.ylim([-80,-50])
	cnt+=1


plt.show()


#Save V_m
pos_filename = 'Vm_ON' + '.cpickle'
f = open(pos_filename, "wb")
cPickle.dump(v_ON, f, protocol=2)
f.close()

pos_filename = 'Vm_OFF' + '.cpickle'
f = open(pos_filename, "wb")
cPickle.dump(v_OFF, f, protocol=2)
f.close()


# Save spikes
pos_filename = 'spike_ON' + '.cpickle'
f = open(pos_filename, "wb")
cPickle.dump(nsON, f, protocol=2)
cPickle.dump(spikeON, f, protocol=2)
f.close()

pos_filename = 'spike_OFF' + '.cpickle'
f = open(pos_filename, "wb")
cPickle.dump(nsOFF, f, protocol=2)
cPickle.dump(spikeOFF, f, protocol=2)
f.close()






