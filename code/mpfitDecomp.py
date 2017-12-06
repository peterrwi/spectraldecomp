import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy import units as u
import scipy.optimize as op
from scipy.interpolate import interp1d
import mpyfit
import config
from functions import *
import time

def prior(theta, priors):
	for i in range(len(theta)):
		if not priors[i][0] < theta[i] < priors[i][1]:
			return np.inf
	return 0.0

def lnlike(theta,args):
	x, y, func_list, p0_len_list = args
	p0 = [0]*len(p0_len_list)
	for i in range(len(p0)):
		indexstart = sum(p0_len_list[i] for i in range(i))
		p0[i] = theta[indexstart:indexstart+p0_len_list[i]]
	
	model = sum(func_list[i](x,p0[i]) for i in range(len(func_list)))
	
	return model - y

def lnprob(theta, args):
	x, y, func_list, priors, p0_len_list = args
	lp = prior(theta,priors)
	deviates = lp + lnlike(theta, (x, y, func_list,p0_len_list))

	return deviates

if __name__ == "__main__":
	datadir = "../data/testspectrum/"
	specname = "testspectrum.txt"
	x = np.loadtxt(datadir+specname)
	x = x[500:1300]

	t0 = time.time()

	components = config.returnComponents()

	key_list = components.keys()
	func_list = [components[key][0] for key in key_list]
	p0_list = [components[key][1] for key in key_list]
	p0_len_list = [len(components[key][1]) for key in key_list]
	prior_list = [components[key][2] for key in key_list]

	p0 = reduce(lambda x,y:x+y,p0_list)
	priors = reduce(lambda x,y:x+y,prior_list)

	# Do the fitting
	print "Fitting using mpyfit..."
	pfit, results = mpyfit.fit(lnprob, p0, (x[:,0],x[:,1],func_list,priors,p0_len_list))
	theta_best = pfit

	theta = [0]*len(func_list)
	for i in range(len(theta)):
		indexstart = sum(p0_len_list[i] for i in range(i))
		theta[i] = theta_best[indexstart:indexstart+p0_len_list[i]]
	for i in range(len(key_list)):
		print key_list[i], theta[i]

	plt.plot(x[:,0],x[:,1],'k')
	plt.plot(x[:,0],sum(func_list[i](x[:,0],theta[i]) for i in range(len(func_list))))
	#plt.plot(x[:,0],150.*x[:,0]**-0.5)
	for i in range(len(func_list)):
		plt.plot(x[:,0],func_list[i](x[:,0],theta[i]))
	plt.show()






