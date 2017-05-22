"""
This function allow us to visualize the result firing rate from a lgn cell that we can choose and then 
visualize the spikes produce by it 
"""
import nest
import numpy as np
import matplotlib.pyplot as plt
import cPickle

#from misc_functions import create_standar_indexes, create_extra_indexes
#from stimuli_functions import sine_grating, ternary_noise 
#from moving_dot_function import moving_dot
#from kernel_functions import create_kernel 
#from analysis_functions import produce_spikes, convolution
#from plot_functions import visualize_firing_rate


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

params = {"V_th" : -59.4 }
neuronON = nest.Create("iaf_chxk_2008", dimensionON)
nest.SetStatus(neuronON, params)

#params = {"V_th" : -59.2 }
neuronOFF = nest.Create("iaf_chxk_2008", dimensionOFF)
nest.SetStatus(neuronOFF, params)

xn = np.arange(0,dimension,1)
nsON = np.zeros((dimensionON,1))
nsOFF = np.zeros((dimensionOFF,1))


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


cnt=1
plt.figure(1)
for count in np.arange(0,dimensionON,1):	
	plt.subplot(Nx,Ny,cnt)
	plt.ylabel("firing rate ON")
	plt.plot(scaled_firing_rateON[:,count])
	plt.ylim([0,100])
	cnt+=1
plt.savefig('figure 1.png')

cnt=1
plt.figure(2)
for count in np.arange(0,dimensionON,1):
	plt.xlabel("Time")
	plt.ylabel("spike ON")	
	y = np.ones_like(spikeON[:nsON[count],count]) 
	y*=cnt
	plt.plot(spikeON[:nsON[count],count],y, '*', label='spikes')
	plt.xlim([0,t_sim])
	cnt+=1
plt.savefig('figure 2.png')

cnt=1
plt.figure(3)
for count in np.arange(0,dimensionON,1):
	plt.subplot(Nx,Ny,cnt)
	plt.plot(v_ON[:,count])
	plt.ylabel('Vm - ON')
	plt.ylim([-80,-50])
	cnt+=1
plt.savefig('figure 3.png')

cnt=1
plt.figure(4)	
for count in np.arange(0,dimensionOFF,1):	
	plt.subplot(Nx,Ny,cnt)
	plt.ylabel("firing rate OFF")
	plt.plot(scaled_firing_rateOFF[:,count])
	plt.ylim([0,100])
	cnt+=1
plt.savefig('figure 4.png')

cnt=1
plt.figure(5)
for count in np.arange(0,dimensionOFF,1): 
		plt.xlabel("Time")
		plt.ylabel("spike OFF")
		y = np.ones_like(spikeOFF[:nsOFF[count],count])
		y+=cnt
		plt.plot(spikeOFF[:nsOFF[count],count],y, '*', label='spikes')
		plt.xlim([0,t_sim])
		cnt+=1
plt.savefig('figure 5.png')		

cnt=1
plt.figure(6)
for count in np.arange(0,Nx*Ny,1):
	plt.subplot(Nx,Ny,cnt)
	plt.plot(v_OFF[:,count])
	plt.ylabel('Vm - OFF')
	plt.ylim([-80,-50])
	cnt+=1
plt.savefig('figure 6.png')

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






