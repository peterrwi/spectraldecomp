import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy import units as u
import scipy.optimize as op
from scipy.interpolate import interp1d
import emcee
import config
from functions import *

def lnprior(theta, priors):
	for i in range(len(theta)):
		if not priors[i][0] < theta[i] < priors[i][1]:
			return -np.inf
	return 0.0

def lnlike(theta,x,y,func_list,p0_len_list):
	p0 = [0]*len(p0_len_list)
	for i in range(len(p0)):
		indexstart = sum(p0_len_list[i] for i in range(i))
		p0[i] = theta[indexstart:indexstart+p0_len_list[i]]
	
	model = sum(func_list[i](x,p0[i]) for i in range(len(func_list)))
	
	return -0.5 * np.sum((model - y)**2)

def lnprob(theta, x, y, func_list, priors,p0_len_list):
	lp = lnprior(theta,priors)
	if not np.isfinite(lp):
		return -np.inf
	chisq = lp + lnlike(theta, x, y, func_list,p0_len_list)
	print chisq
	return chisq

if __name__ == "__main__":
	datadir = "../data/testspectrum/"
	specname = "testspectrum.txt"
	x = np.loadtxt(datadir+specname)
	x = x[500:1300]

	components = config.returnComponents()

	key_list = components.keys()
	func_list = [components[key][0] for key in key_list]
	p0_list = [components[key][1] for key in key_list]
	p0_len_list = [len(components[key][1]) for key in key_list]
	prior_list = [components[key][2] for key in key_list]

	p0 = reduce(lambda x,y:x+y,p0_list)
	priors = reduce(lambda x,y:x+y,prior_list)

	ndim,nwalkers = len(p0),1000

	emceep0 = np.vstack([np.random.normal(p0[i],1,nwalkers) for i in range(len(p0))]).T

	sampler = emcee.EnsembleSampler(
		nwalkers,
		ndim,
		lnprob,
		args=(x[:,0],x[:,1],func_list,priors,p0_len_list)
		)

	sampler.run_mcmc(emceep0,1000)

	samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

	theta_best = np.percentile(samples,50,axis=0)
	theta = [0]*len(func_list)
	for i in range(len(theta)):
		indexstart = sum(p0_len_list[i] for i in range(i))
		theta[i] = theta_best[indexstart:indexstart+p0_len_list[i]]
	for i in range(len(key_list)):
		print key_list[i], theta[i]

	plt.plot(x[:,0],x[:,1],'k')
	plt.plot(x[:,0],sum(func_list[i](x[:,0],theta[i]) for i in range(len(func_list))))
	plt.plot(x[:,0],150.*x[:,0]**-0.5)
	for i in range(len(func_list)):
		plt.plot(x[:,0],func_list[i](x[:,0],theta[i]))
	plt.show()

	#import corner
	#fig = corner.corner(samples)#, labels=["$p1$", "$p2$"])
	#fig.savefig("triangle.png")

	exit()














	pfit, results = mpyfit.fit(mpfitlnprob, p0, (x[:,0],x[:,1],func_list,priors))
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


