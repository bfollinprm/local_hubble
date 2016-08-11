import sys
sys.path.append("/Users/follin/software/likelihoods/plc-2.0/lib/python2.7/site-packages")
#sys.path.append("/Users/follin/Documents/projects/p_of_k_test/cosmoslik/lib/python2.7/site-packages")

from cosmoslik import param_shortcut, lsum, get_plugin, SlikDict, SlikPlugin, Slik


from numpy import logspace, savetxt, inf, log10, exp, log, zeros
from scipy.stats import lognorm

param = param_shortcut('start', 'scale')

class main(SlikPlugin):
	'''
	Runs a chain with arbitrary power spectrum
	'''


	def __init__(self, logkstep = 0.025, kmin = 0.0001, kmax = 0.1):
		super(SlikPlugin,self).__init__()
		self.mle = {'lnl': inf}
		self.cosmo = get_plugin('models.cosmology')(
		    logA = param(3.3),
		    ns = param(0.96),
		    ombh2 = param(0.0221),
		    omch2 = param(0.12),
		    tau = param(0.13,min=0),
		    theta = param(0.010413),
		    omnuh2 = 0.000645,
		    massless_neutrinos=3.046, #param(3,.2),
		            )

		self.kvals = logspace(log10(kmin), log10(kmax), num = (log10(kmax) - log10(kmin))/logkstep)
		self.kstar = 0.005
		self.logkstep = logkstep
		for k in self.kvals:
			name = 'A%1.3d'%log10(k)
			self.cosmo[name.replace('.','')] = param(0, scale = .01)


		self.cosmo.Tcmb = 2.7255

		### CMB Likelihoods
		self.camspec = get_plugin('likelihoods.clik')(
		    clik_file='/Users/follin/software/likelihoods/plc-2.0/hi_l/plik_lite/plik_lite_v18_TT.clik',
		    A_Planck = param(1, min = 0, scale = .00025,gaussian_prior = (1,0.00025))
		)


		self.lowl = get_plugin('likelihoods.clik')(
		  clik_file='/Users/follin/software/likelihoods/plc-2.0/low_l/commander/commander_rc2_v1.1_l2_29_B.clik',
		  A_planck = self.camspec.A_Planck
		)

		### Forward Modellers
		self.get_cmb = get_plugin('models.classy')()
		
		# self.get_cmb = get_plugin('models.pico')(datafile = 'pico3_tailmonty_v34.dat')
		self.bbn = get_plugin('models.bbn_consistency')()
		self.deriver = get_plugin('models.cosmo_derived')()
		self.priors = get_plugin('likelihoods.priors')(self)


		self.sampler = get_plugin('samplers.metropolis_hastings')(
		     self,
		     num_samples=60000,
		     output_file='../vary_pk.chain',
		     proposal_cov='proposal.covmat',
		     proposal_scale=0.5,
		     print_level=0,
		     output_extra_params=['cosmo.Yp', 'camspec.A_Planck']
		)





	def __call__(self):
    	### Construct primordial power spectrum
		# ksamples = logspace(-6, 1, num = 7.0*200)
		# pk0 = exp(self.cosmo.logA) * (ksamples/self.kstar)**(self.cosmo.ns - 1.0)
		# dpk = zeros(ksamples.size)
		# for k in self.kvals:
		# 	name = 'A%1.3d'%log10(k)
		# 	dpk += (self.cosmo[name.replace('.','')]*exp( - 0.5 * (log(ksamples) - log(k))**2/self.logkstep**2))
		# pk = (1 +dpk)*pk0
		# savetxt('pk_table.dat',zip(ksamples, pk))

		self.cosmo.Yp = self.bbn(**self.cosmo)
		self.deriver.set_params(**self.cosmo)
		self.cosmo.H0 = self.deriver.theta2hubble(self.cosmo.theta)
		

		self.cmb_result = self.get_cmb(**self.cosmo)
		#self.cmb_result = self.get_cmb(force = True, **self.cosmo)
		if (self.cosmo.H0 -73.24)**2/1.74**2 > 4: 
			return (self.cosmo.H0 -73.24)**2/1.74**2 * 100000
		else:

			self.lnl =  lsum(lambda: self.camspec(self.cmb_result),
		            #lambda: self.lowl(self.cmb_result),
		            lambda: self.priors(self),
		            #lambda: 0.5 * (self.cosmo.H0 - 73.24)**2/1.74**2
		            )
			print 'H_0 = %3.3f'%self.cosmo.H0, 'lnl = %3.3f'%self.lnl
			return self.lnl

if __name__=='__main__':
     #run the chain
     for _ in Slik(main()).sample(): 
            pass
