from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.interpolate import interp1d
import mpyfit
from functions import *
import time
import sys
import re

rcParams['text.usetex']=True
rcParams['text.latex.unicode']=True

class Component():
	def __init__(self,funcname):
		self.func = eval(funcname+"()")
		self.nparms = self.func.nparms
		self.index = None
		self.params = None
		self.prior = None
		self.texname = None
		self.fixed = [False]*self.nparms

def deviates(theta,args):
	x,y,func,components = args	
	model = func(x,theta,components)
	deviates = y - model
	return deviates

def buildFunction(x,theta,components):
	keys = sort(components.keys())
	func_list = [components[key].func for key in keys]
	p0_list = [components[key].params for key in keys]
	p0_len_list = [len(components[key].params) for key in keys]
	prior_list = [components[key].prior for key in keys]
	fixed_list = [components[key].fixed for key in keys]

	p0 = reduce(lambda x,y:x+y,p0_list)
	priors = reduce(lambda x,y:x+y,prior_list)
	fixed = reduce(lambda x,y:x+y,fixed_list)
	limited = [[True,True]]*len(p0)

	p0new = []
	args = []
	i_counter = 0
	for i in range(len(fixed)):
		if fixed[i]:
			args.append(p0[i])
		else:
			args.append(theta[i_counter])
			p0new.append(p0[i])
			i_counter+=1

	args_array = [0]*len(func_list)
	for i in range(len(args_array)):
		startindex = sum(p0_len_list[j] for j in range(i))
		endindex = sum(p0_len_list[j] for j in range(i+1))
		args_array[i] = args[startindex:endindex]

	f = sum(func_list[i](x,args_array[i]) for i in range(len(func_list)))

	return f

def readConfigFile(configfile):
	with open(configfile,'r') as f:
		lines = f.read().splitlines() 
	linenumbers = []
	startfile = None
	l = 0
	# Go through lines one by one and execute commands in order
	while l < len(lines):
		if re.match("specloc",lines[l]):
			specloc = lines[l].split()[1]
			x = np.loadtxt(specloc)
			x = x[500:1300]
			print "Loaded spectrum from %s" % specloc

		if re.match("funcfile",lines[l]):
			funcfile = lines[l].split()[1]
			# Get the functions from funcfile
			with open(funcfile,'r') as f:
				funcs = f.read().splitlines()
			for j in range(len(funcs)):
				if re.match("^[0-99]\)",funcs[j]):
					linenumbers.append(j)
			print "Loaded list of available functions from %s" % funcfile

		if re.match("constraintfile",lines[l]):
			constraintfile = lines[l].split()[1]
			print "Loaded constraints from %s" % constraintfile

		if re.match("startfile",lines[l]):
			startfile = lines[l].split()[1]
			print "Using the startfile %s" % startfile

		if re.match("fit",lines[l]) or re.match("grid",lines[l]):
			if re.match("fit",lines[l]):
				funcnumbers = lines[l].split()[1].split(',')
				print "Starting a fitting sequence with functions ", funcnumbers
			elif re.match("grid",lines[l]):
				funcnumbers = lines[l].split()[1].split(',')
				gridfuncs = lines[l+1].split(',')
				gridparms = map(int,lines[l+2].split(','))
				print "Starting a grid search sequence with functions ", funcnumbers
				print "Doing grid search on functions ", gridfuncs, " parameters ",gridparms
				gridrange = map(
					lambda x:map(float,x.split(',')),
					lines[l+3].split(';')
					)
			
			# Go through following lines to lead in any extra commands/options
			t=1
			fixed, plotting, outfile = [], False, None
			while t > 0:
				if re.match("fixed",lines[l+t]):
					tmp = lines[l+t].split()[1]
					fixed = np.array(map(
						lambda x:map(int,x.split(',')),
						tmp.split(';')
						))
					print "Fixing parameters:"
					for i in range(len(fixed)):
						print "  Function %i, parameter %i" % (fixed[i,0],fixed[i,1])
				if re.match("outfile",lines[l+t]):
					outfile = lines[l+t].split()[1]
					print "Saving results to %s" % outfile
				if re.match("startfile",lines[l+t]):
					startfile = lines[l+t].split()[1]
					print "Using the startfile %s" % startfile
				if re.match("plotfit",lines[l+t]):
					plotting = True
				if re.match("end",lines[l+t]):
					t=-1e9
				t+=1

			components = {}
			for j in linenumbers:
				if funcs[j][:-1] in funcnumbers:
					funcname = funcs[j+2]
					paramguess = map(float,funcs[j+3].split(','))
					paramlims = map(
						lambda x:map(float,x.split(',')),
						funcs[j+4].split(';')
						)
					texname = funcs[j+5]
					strname = funcs[j+1]
					
					index = funcs[j][:-1]
					components[index] = Component(funcname)
					components[index].params = paramguess
					components[index].prior = paramlims
					components[index].texname = texname
					components[index].name = strname
					for i in range(len(fixed)):
						if index == str(fixed[i,0]):
							components[index].fixed[fixed[i,1]] = True

			if re.match("fit",lines[l]):
				pfit,results = executeFit(components,startfile,x)
			elif re.match("grid",lines[l]):
				pfit,results = gridSearch(components,startfile,x,gridfuncs,gridparms,gridrange)

			if outfile != None:
				saveStartFile(outfile,pfit)
			
			for key in funcnumbers:
				print components[key].name+": ",pfit[key]
			if plotting:
				plotFit(x,components,pfit,saveloc=False)

		if re.match("quit",lines[l]):
			l = 1e9
		l += 1
		
	return pfit

def readStartFile(startfile,components):
	if startfile == None:
		return components
	print "Using the startfile %s" % startfile
	print "Loading parameters: "
	with open(startfile,'r') as f:
		for l in f.readlines():
			pars = l.split(',')
			component_name = pars[0]
			print component_name
			components[component_name].params = map(float,pars[1:])
	return components

def saveStartFile(outfile,pfit):
	with open(outfile,'w') as f:
		for key in sort(pfit.keys()):
			f.write(key+",")
			f.write(",".join(map(str,pfit[key])))
			f.write("\n")

def executeFit(components,startfile,x):
	components = readStartFile(startfile,components)
	keys = sort(components.keys())

	pfit,results = fitFunction(x,components)
	return pfit,results

def gridSearch(components,startfile,x,gridfuncs,gridparms,gridrange):
	Ndims = len(gridfuncs)

	components = readStartFile(startfile,components)
	keys = sort(components.keys())

	dims = [int(gridrange[i][2]) for i in range(Ndims)]
	ax = [0]*Ndims
	for i in range(Ndims):
		ax[i] = np.linspace(gridrange[i][0],gridrange[i][1],gridrange[i][2])
	mesh = np.meshgrid(*ax)

	gridresults = []
	for j in range(len(mesh[0].ravel())):
		for i in range(Ndims):
			components[gridfuncs[i]].params[gridparms[i]] = mesh[i].ravel()[j]
		pfit,results = fitFunction(x,components)
	
		print "chi^2: ",results['orignorm']," --> ",results['bestnorm']
		gridresults.append([results['bestnorm'],pfit,results])

	gridresults = np.array(gridresults)
	bestrun = argmin(gridresults[:,0])
	
	return gridresults[bestrun,1],gridresults[bestrun,2]

def fitFunction(x,components):
	# Do the fitting
	print "Fitting using mpyfit..."
	keys = sort(components.keys())
	func_list = [components[key].func for key in keys]
	p0_list = [components[key].params for key in keys]
	p0_len_list = [len(components[key].params) for key in keys]
	prior_list = [components[key].prior for key in keys]
	fixed_list = [components[key].fixed for key in keys]

	p0 = reduce(lambda x,y:x+y,p0_list)
	priors = reduce(lambda x,y:x+y,prior_list)
	fixed = reduce(lambda x,y:x+y,fixed_list)
	limited = [[True,True]]*len(p0)

	tied = [False]*len(p0)
	tied[19] = "3 * p[16]"
	
	parinfo = [{} for i in range(len(p0))]
	for i in range(len(p0)):
		parinfo[i]['limited'] = tuple(limited[i])
		parinfo[i]['limits'] = tuple(priors[i])
		parinfo[i]['fixed'] = fixed[i]
		parinfo[i]['tied'] = tied[i]

	p0new = []
	i_counter = 0
	for i in range(len(fixed)):
		if not fixed[i]:
			p0new.append(p0[i])
			i_counter+=1

	pfit, results = mpyfit.fit(deviates, p0new, (x[:,0],x[:,1],buildFunction,components))

	pfit_complete = []
	i_counter = 0
	for i in range(len(fixed)):
		if fixed[i]:
			pfit_complete.append(p0[i])
		else:
			pfit_complete.append(pfit[i_counter])
			i_counter += 1
	pfit = pfit_complete

	# Put back into dictionary form
	pbest = {}
	for j in range(len(keys)):
		pbest[keys[j]] = pfit[sum(p0_len_list[k] for k in range(j)):sum(p0_len_list[k] for k in range(j+1))]
	return pbest,results

def plotFit(x,components,pfit,saveloc,showtag=True):
	keys = sort(components.keys())
	func_list = [components[key].func for key in keys]
	theta = [pfit[key] for key in keys]
	tex_list = [components[key].texname for key in keys]
	# Plotting values
	figwidth = 13.
	figheight = 6.

	_label_fontparams = {'fontsize':14}
	_axis_fontparams = {'fontsize':14}
	_tick_fontparams = {'fontsize':10}

	_lpadin = 0.9
	_rpadin = 0.1
	_bpadin = 0.5
	_tpadin = 0.1
	_wpadin = 0.0
	_hpadin = 0.0

	_lpad = _lpadin/figwidth
	_rpad = _rpadin/figwidth
	_hpad = _hpadin/figheight
	_tpad = _tpadin/figheight
	_bpad = _bpadin/figheight
	_wpad = _wpadin/figwidth

	_spheight_t = 0.7
	_spheight_b = 1. - _spheight_t - _bpad - _tpad - _hpad
	_spwidth = 1. - _lpad - _rpad

	plt.figure(1,figsize=(figwidth,figheight))
	ax1 = plt.axes((
		_lpad,
		1. - _tpad - _spheight_t,
		_spwidth,
		_spheight_t
		))
	plt.plot(x[:,0],x[:,1],'k')
	plt.plot(
		x[:,0],
		sum(func_list[i](x[:,0],theta[i]) for i in range(len(func_list))),
		color='b',
		label="${\\rm Full~model}$"
		)
	#plt.plot(x[:,0],150.*x[:,0]**-0.5)
	for i in range(len(func_list)):
		plt.plot(x[:,0],func_list[i](x[:,0],theta[i]),label=tex_list[i])
	#plt.xlabel("${\\rm Wavelength~(\\AA)}$")
	plt.ylim(-1,int(max(x[:,1]))+1)
	plt.yticks(range(int(max(x[:,1]))+2))
	ax1.set_xticklabels("")
	plt.ylabel("${\\rm Flux~(arbitrary)}$",**_label_fontparams)
	plt.legend(loc='upper left',ncol=2,frameon=False)

	ax2 = plt.axes((
		_lpad,
		1. - _tpad - _spheight_t - _hpad - _spheight_b,
		_spwidth,
		_spheight_b
		))
	plt.plot(x[:,0],(sum(func_list[i](x[:,0],theta[i]) for i in range(len(func_list))) - x[:,1])/x[:,1],'k')
	plt.axhline(0,ls="--",color='k')
	plt.xlabel("${\\rm Wavelength~(\\AA)}$",**_label_fontparams)
	plt.ylabel("$({\\rm Model} - {\\rm Data})/{\\rm Data}$",**_label_fontparams)
	plt.ylim(-0.17,0.17)
	if saveloc != False:
		plt.savefig(saveloc)
	if showtag:
		plt.show()

def main():
	configfile = sys.argv[1]
	outfile = configfile.split('.')[0]+'.out'
	pfit = readConfigFile(configfile)
	saveStartFile(outfile,pfit)

if __name__ == "__main__":
	main()
	




