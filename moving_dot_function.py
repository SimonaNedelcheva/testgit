import numpy as np
from math import cos, sin

def moving_dot(lx, ly, dx, dy, N_stimuli, dt_stimuli):

	#moving dot stimuli
	Nx = int(lx / dx)
	Ny = int(ly / dy)
	x = np.arange(-lx/2, lx/2, dx)
	y = np.arange(-ly/2, ly/2, dy)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros((Nx, Ny), dtype=float)
	Z += 0.5
#	t = np.arange(0, N_stimuli * dt_stimuli, dt_stimuli)
	stimuli = np.zeros((int(N_stimuli*dt_stimuli), Nx, Ny), dtype=float)

	xc0 = -2.0
	yc0 = -2.0
	dd = 0.015
	rd = 0.1
	fi = np.linspace(0, 2 * np.pi, 100)
	r = np.linspace(0, rd, 100) #circles radius
	xc = np.zeros(int(N_stimuli*dt_stimuli), dtype=float) #dot center x
	yc = np.zeros(int(N_stimuli*dt_stimuli), dtype=float) #dot center y
	xc[0:] = xc0
	yc[0:] = yc0

	stim_step = int(dt_stimuli*5)
	stim_t = np.arange(0, int(N_stimuli*dt_stimuli), stim_step).astype(int)
	rk = np.arange(0, len(r), 1).astype(int)
	fik=np.arange(0, len(fi), 1).astype(int)
	ns = 0
	for k in stim_t: #1:stim_step:N_stimuli*dt_stimuli
		for kr in rk: #1:size(r,2)
      			for kfi in fik: #1:size(fi,2)
	       	    		dpx = X[0,:] - xc[ns] - r[kr] * sin(fi[kfi])
            			dix = np.argmin(abs(dpx))
	       			dpy = Y[:,0] - yc[ns] - r[kr] * cos(fi[kfi])
	       			diy = np.argmin(abs(dpy))
	       			Z[dix, diy] = 0.0 #1.0
		xc[ns+1] = xc[ns] + dd
        	yc[ns+1] = yc[ns] + dd
        	ns += 1
            	s = np.arange(k, k+stim_step, 1).astype(int)
            	for index in s:
                	stimuli[index, ...] = Z
            	Z[0:, 0:] = 0.5

	return stimuli
