import numpy as np
from scipy.interpolate import interp1d
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
import time

def log_rebin(lamRange, spec, oversample=False, velscale=None, flux=False):
    """
    From https://github.com/moustakas/impy/blob/master/lib/ppxf/ppxf_util.py

    Logarithmically rebin a spectrum, while rigorously conserving the flux.
    Basically the photons in the spectrum are simply redistributed according
    to a new grid of pixels, with non-uniform size in the spectral direction.
    
    When the flux keyword is set, this program performs an exact integration 
    of the original spectrum, assumed to be a step function within the 
    linearly-spaced pixels, onto the new logarithmically-spaced pixels. 
    The output was tested to agree with the analytic solution.
    """
    lamRange = np.asarray(lamRange)
    if len(lamRange) != 2:
        raise ValueError('lamRange must contain two elements')
    if lamRange[0] >= lamRange[1]:
        raise ValueError('It must be lamRange[0] < lamRange[1]')
    s = spec.shape
    if len(s) != 1:
        raise ValueError('input spectrum must be a vector')
    n = s[0]
    if oversample:
        m = int(n*oversample)
    else:
        m = int(n)

    dLam = np.diff(lamRange)/(n - 1.)        # Assume constant dLam
    lim = lamRange/dLam + [-0.5, 0.5]        # All in units of dLam
    borders = np.linspace(*lim, num=n+1)     # Linearly
    logLim = np.log(lim)

    c = 299792.458                           # Speed of light in km/s
    if velscale is None:                     # Velocity scale is set by user
        velscale = np.diff(logLim)/m*c       # Only for output
    else:
        logScale = velscale/c
        m = int(np.diff(logLim)/logScale)    # Number of output pixels
        logLim[1] = logLim[0] + m*logScale

    newBorders = np.exp(np.linspace(*logLim, num=m+1)) # Logarithmically
    k = (newBorders - lim[0]).clip(0, n-1).astype(int)

    specNew = np.add.reduceat(spec, k)[:-1]  # Do analytic integral
    specNew *= np.diff(k) > 0    # fix for design flaw of reduceat()
    specNew += np.diff((newBorders - borders[k])*spec[k])

    if not flux:
        specNew /= np.diff(newBorders)

    # Output log(wavelength): log of geometric mean
    logLam = np.log(np.sqrt(newBorders[1:]*newBorders[:-1])*dLam)

    return specNew, logLam, velscale

def gaussian_filter1d(spec, sig):
    """
    From https://github.com/moustakas/impy/blob/master/lib/ppxf/ppxf_util.py

    Convolve a spectrum by a Gaussian with different sigma for every
    pixel, given by the vector "sigma" with the same size as "spec".
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating  template library for SDSS data, this implementation
    is 60x faster than the naive loop over pixels.
    """

    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = np.sum(a*gau, 0)

    return conv_spectrum

def convolveGauss(x,y,sig,pad):
	conv = np.zeros(len(x))
	dx = x[1]-x[0]
	xgauss = np.arange(-dx*pad,dx*pad,dx)
	ygauss = Gauss().func(xgauss,(1.,0.,sig))

	for i in range(len(x)):
		l0 = x[i]
		minsum = max(0,i-pad)
		maxsum = min(len(x),i+pad)
		xx = np.array([x[j] for j in range(minsum,maxsum)])
		yy = np.array([y[j] * ygauss[j-i+pad] for j in range(minsum,maxsum)])
		gaussian_trimmed = np.array([ygauss[j-i+pad] for j in range(minsum,maxsum)])
		conv[i] = np.sum(yy)/np.sum(gaussian_trimmed)
	return conv

def logBroaden(x,dv):
	res=2.
	intres=10
	
	# First re-bin to a log-linear wavelength scale
	#t0 = time.time()
	Flogl,logl,velscale = log_rebin([x[0,0],x[-1,0]],x[:,1])
	#print "log_rebin: ",time.time()-t0

	# Do the broadening
	#t0 = time.time()
	#dv = 100.
	Floglbroad = gaussian_filter1d(Flogl,dv/velscale)
	#print "Broadening: ",time.time()-t0
	Flogl = Floglbroad

	#plt.figure()
	#plt.plot(logl[300:500],Flogl[300:500])
	#plt.plot(logl[300:500],Floglbroad[300:500])
	#plt.show()
	#exit()

	# Re-bin back to linear wavelength scale
	xinterp = interp1d(logl,Flogl)
	l = x[:,0].copy()
	Fl = np.zeros(len(l))
	dl = (l[-1]-l[0])/len(l)
	for i in range(1,len(l)-1):
		xx = np.linspace(l[i]-dl/2.,l[i]+dl/2.,intres,endpoint=True)
		logxx = np.log(xx)
		logyy = xinterp(logxx)
		Fl[i] = np.trapz(logyy,logxx)/(logxx[-1]-logxx[0])
	
	return np.vstack([l,Fl]).T

class GaussHermite:
	def __init__(self):
		#self.x = x
		#self.theta = theta
		self.nparms = 8

	def __call__(self,x,theta):
		return self.func(x,theta)
	
	def func(self,x,theta):
		gamma, V, sigma, h0, h1, h2, h3, h4 = theta
		h = [h0, h1, h2, h3, h4]
		w = (x - V)/sigma
		gausspart = 1./np.sqrt(2.*np.pi)*np.exp(-0.5*w**2)/sigma
		return gamma * gausspart * self.hermitePolynomial(w,h)

	def hermitePolynomial(self,x,h):
		return h[0] + h[1]*np.sqrt(2)*x + h[2]/np.sqrt(2)*(2.*x**2 - 1.) + \
			h[3]/np.sqrt(6)*(2.*np.sqrt(2)*x**3 - 3.*np.sqrt(2)*x) + \
			h[4]/np.sqrt(24)*(4.*x**4 - 12.*x**2 + 3.)

class Gauss:
	def __init__(self):
		#self.x = x
		#self.theta = theta
		self.nparms = 3

	def __call__(self,x,theta):
		return self.func(x,theta)
	
	def func(self,x,theta):
		A, mu, sigma = theta
		return A/(np.sqrt(2.*np.pi) * sigma) * np.exp(-0.5 * ((x - mu)/sigma)**2)

class Offset:
	def __init__(self):
		self.nparms = 1

	def __call__(self,x,theta):
		return self.func(x,theta)
	
	def func(self,x,theta):
		offset = theta
		return x*0+offset

class PowerLaw:
	def __init__(self):
		#self.x = x
		#self.theta = theta
		self.nparms = 2

	def __call__(self,x,theta):
		return self.func(x,theta)
	
	def func(self,x,theta):
		A, alpha = theta
		return A * x**alpha / (5000**alpha)

class Starlight:
	def __init__(self):
		self.nparms = 3
		#self.dv = 100.
		#self.stars_interp = self.starlight_interpolate(self.dv)
		self.memoize = {"dv":{}}
	def __call__(self,x,theta):
		return self.func(x,theta)
	
	def func(self,x,theta):
		A, z, dv = theta
		if dv in self.memoize["dv"].keys():
			self.stars_interp = self.memoize["dv"][dv]
		else:
			self.stars_interp = self.starlight_interpolate(dv)
			self.memoize["dv"][dv] = self.stars_interp
		return A * self.stars_interp(x/(1.+z))

	def starlight_interpolate(self,dv):
		# Bruzual & Charlot (2003) templates
		bc03_dir = "../data/bc03_2016/templates/"
		templatename = "ssp_11Gyr_z008.spec.gz"
		template = np.loadtxt(bc03_dir+templatename)[1500:2800]

		template = logBroaden(template,dv)
		template = np.vstack([np.vstack([np.array([0,0]),template]),np.array([8000,0])])
		f_interp = interp1d(template[:,0],template[:,1])
		return f_interp

class FeII:
	def __init__(self):
		self.nparms = 6
		self.dv = 100.
		self.feii_interp = self.starlight_interpolate(self.dv/3.e5)

	def __call__(self,x,theta):
		return self.func(x,theta)
	
	def func(self,x,theta):
		A1,A2,A3,A4,A5,z = theta
		A = [A1,A2,A3,A4,A5]
		return sum(A[i] * self.feii_interp[i](x/(1.+z)) for i in range(len(A)))

	def feII_interpolate(self,dv):
		feii_dir = "../data/FeII_template_4000_5500/700FeII/"
		templatenames = ["fe_f.txt","fe_g.txt","fe_IZw1.txt","fe_p.txt","fe_s.txt"]
		f_interp = [0]*len(templatenames)
		for i in range(len(templatenames)):
			template = np.loadtxt(feii_dir+templatenames[i])
			#template = logBroaden(template,dv)
			template = np.vstack([np.vstack([np.array([0,0]),template]),np.array([8000,0])])
			f_interp[i] = interp1d(template[:,0],template[:,1])
		return f_interp

def testing():
	testclass = eval("Starlight()")
	print testclass
	x = np.arange(4700,5300,1)
	theta = [0.7,0.0417,100.]
	f = testclass.func
	print f
	print f(x,theta)

if __name__=="__main__":
	testing()










