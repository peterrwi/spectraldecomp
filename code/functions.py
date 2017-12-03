import numpy as np
from scipy.interpolate import interp1d

def hermitePolynomial(x,h):
	return h[0] + h[1]*np.sqrt(2)*x + h[2]/np.sqrt(2)*(2.*x**2 - 1.) + \
			h[3]/np.sqrt(6)*(2.*np.sqrt(2)*x**3 - 3.*np.sqrt(2)*x) + \
			h[4]/np.sqrt(24)*(4.*x**4 - 12.*x**2 + 3.)

def gaussHermite(x,theta):
	gamma, V, sigma, h0, h1, h2, h3, h4 = theta
	h = [h0, h1, h2, h3,h4]
	w = (x - V)/sigma
	gausspart = 1./np.sqrt(2.*np.pi)*np.exp(-0.5*w**2)/sigma
	return gamma * gausspart * hermitePolynomial(w,h)

def gauss(x,theta):
	A, mu, sigma = theta
	return A/(np.sqrt(2.*np.pi) * sigma) * np.exp(-0.5 * ((x - mu)/sigma)**2)

def offset(x,theta):
	offset = theta
	return x*0+offset

def powerLaw(x,theta):
	A, alpha = theta
	return A * x**alpha / (5000**alpha)

def starlight_interpolate():
	# Bruzual & Charlot (2003) templates
	bc03_dir = "../data/bc03_2016/templates/"
	templatename = "ssp_11Gyr_z008.spec.gz"
	template = np.loadtxt(bc03_dir+templatename)[300:6000]
	template = np.vstack([np.vstack([np.array([0,0]),template]),np.array([8000,0])])
	f_interp = interp1d(template[:,0],template[:,1])
	return f_interp

def starlight(x,theta):
	A, z = theta
	return A * stars_interp(x/(1.+z))

def feII_interpolate():
	feii_dir = "../data/FeII_template_4000_5500/700FeII/"
	templatenames = ["fe_f.txt","fe_g.txt","fe_IZw1.txt","fe_p.txt","fe_s.txt"]
	f_interp = [0]*len(templatenames)
	for i in range(len(templatenames)):
		template = np.loadtxt(feii_dir+templatenames[i])
		template = np.vstack([np.vstack([np.array([0,0]),template]),np.array([8000,0])])
		f_interp[i] = interp1d(template[:,0],template[:,1])
	return f_interp

def feII(x,theta):
	A1,A2,A3,A4,A5,z = theta
	A = [A1,A2,A3,A4,A5]
	return sum(A[i] * feii_interp[i](x/(1.+z)) for i in range(len(A)))

stars_interp = starlight_interpolate()
feii_interp = feII_interpolate()






