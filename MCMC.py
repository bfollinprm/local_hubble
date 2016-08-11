
# coding: utf-8

# In[2]:

from numpy import polyfit
from astropy.utils.data import download_file
from astropy.utils.data import Conf 
Conf.remote_timeout = 10
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'pylab')
get_ipython().magic(u'matplotlib inline')
from cosmoslik import get_plugin, param, SlikPlugin, SlikDict, run_chain
from cosmoslik.utils import load_chain
rc('text', usetex = True)
import pandas as pd
#from progress.bar import Bar


# In[45]:

class Hubble_Chain(SlikDict):
    def __init__(self, quad = True, cubic = False, numsamples = 50000):
        #self.bar = Bar('Samples', max = numsamples)
        
        self.cepheids, self.sne = get_data()
        self.cepheids = self.cepheids.loc[self.cepheids.Flag == '-',:]

        self.fields = set(self.cepheids.Field)
        self.params = SlikDict()
        self.numsamples = numsamples
        for name in set(self.cepheids.Field):
            self.params[('\delta\mu_p_'+name)] = param(start = 2.5, scale = .04)
        self.params['\delta\mu_p_n4258'] = 0
        self.params['z_pn4258'] = param(start = 26.27, scale = 0.08)
        self.params.b = param(start = -2.98, scale = .06)
        self.params.Z = param(start = -0.240, scale = .09)
        self.params.color_bias = 0#param(start = 0 , scale = 0.04, gaussian_prior= (0,1))
        self.params.SN_mag_4258 = param(start = 10.25, scale = .05)
        self.params.geo_dist = param(start = 7.6, scale  = 0.32, gaussian_prior = (7.6, 0.32))
        self.params['a_nu'] = param(start = 0.698,scale = 0.00225, gaussian_prior = (0.698,0.00225))      
        
        self.sampler = get_plugin('samplers.metropolis_hastings')(self,
                                                                  num_samples=self.numsamples,
                                                                  proposal_scale = 1,
                                                                  print_level = 1,  
                                                                  output_file = 'chain.chain',
                                                                 proposal_cov = 'proposal.covmat',
                                                                 proposal_update = True,
                                                                 output_extra_params=['params.H0']
                                                                 )

        self.priors = get_plugin('likelihoods.priors')(self)
        
        
        
        
    def ceph_mag(self, cepheids, **params):
        
        cepheids.loc[:,'FittedMag'] = (params['z_pn4258'] 
                        + params['b'] * log10(cepheids.Period) 
                        + params['Z'] * (cepheids.ObyH - mean(cepheids.ObyH))##8.9)#mean(cepheids.loc[cepheids.Field == 'n4258', 'ObyH'])) 
                        + 0.410 * cepheids.VtoI
                       )
        for field in (self.fields):
            cepheids.loc[cepheids.Field == field,'FittedMag'] = (cepheids.loc[cepheids.Field == field, 'FittedMag'] 
                                                                 + params['\delta\mu_p_'+field]
                                                                )
        return cepheids.FittedMag.values
    
    def sne_mag(self,sne,**params):

        sne.loc[:, 'FittedMag'] = params['SN_mag_4258']
        for field in self.fields:
            sne.loc[sne.Host == field, 'FittedMag'] += params['\delta\mu_p_'+field]

        return sne.FittedMag.values
    
    
    def __call__(self):
        self.params['\mu_geometric'] = 5 * log10(self.params.geo_dist) + 25
        self.params.H0 = 10**((self.params.SN_mag_4258 - 
                               self.params['\mu_geometric'] 
                               + 5 * self.params['a_nu'] + 25.0)
                              /5.0)
        cephlnl = sum((self.cepheids.F160Wmag - self.ceph_mag(self.cepheids, **self.params))**2/self.cepheids.e_F160Wmag**2)
        #cephlnl = sum((log10(self.cepheids.period) - self.fitted_ceph_period()[0])**2/self.fitted_ceph_period()[1]**2)
        snelnl = sum((self.sne.m0_viPlus5a_v - 5 * self.params['a_nu'] - self.sne_mag(self.sne, **self.params))**2/self.sne.sigma**2)
        priors = sum(self.priors(self))
        #self.bar.next()
        return priors + (cephlnl + snelnl)
        
    
def get_data():
    cepheid_table = download_file(
        'http://iopscience.iop.org/0004-637X/730/2/119/suppdata/apj383673t2_mrt.txt', 
        cache = True)
    cepheids = pd.read_csv(cepheid_table,
                       delim_whitespace = True,
                       skiprows = 39,
                       names = (['Field', 'RAdeg', 
                                 'DEdeg', 'ID', 
                                 'Period', 'VtoI', 
                                 'F160Wmag', 'e_F160Wmag',
                                 'Offset', 'Bias', 
                                 'IMrms', 'ObyH', 'Flag']
                               )
                      )

    cepheids=cepheids.fillna(value = '-')
    ### SNe table
    Sne_table = download_file(
        'http://iopscience.iop.org/0004-637X/730/2/119/suppdata/apj383673t3_ascii.txt',
        cache = True)
    Sne = pd.read_csv(Sne_table, 
                   
                  delim_whitespace=True, 
                  skiprows = [0,1,2,3,4,13,15],
                  names = (['Host', 'junk','Sn1A',
                            'Filters', 'm0_viPlus5a_v',
                            'sigma', 'DeltaMu_0','e_DeltaMu_0',
                            'mu_0_Best', 'e_mu_0_Best'
                          ])
                 )
    Sne.loc[:,'e_DeltaMu_0'] = (Sne.loc[:,'e_DeltaMu_0'].apply(str).str.replace('\(|\)','')).astype('float')
    Sne.loc[:,'e_mu_0_Best'] = (Sne.loc[:,'e_mu_0_Best'].apply(str).str.replace('\(|\)','')).astype('float')
    
    return cepheids, Sne


# In[54]:

std_chain = run_chain(Hubble_Chain, kwargs ={'quad':False, 'numsamples': 50000}).burnin(0)
print std_chain.acceptance()
x =  zip(std_chain.params(),std_chain.mean(),std_chain.std())
for i in x:
    print i


# In[68]:

std_chain = load_chain('chain.chain').burnin(2000).join()
rc('text', usetex = False)
std_chain.plot()
std_chain.savecov('proposal.covmat')


# In[69]:

std_chain.like1d('params.H0', linewidth = 4, label = 'standard', nbins = 25)
legend()
#cubic_chain.like1d('params.H0', color = 'r', linewidth = 4)
ylabel('Likelihood', fontsize = 16)
xlabel(r'Local Hubble Rate $[{\rm km/Mpc}]^{-1}$', fontsize = 16)


# In[70]:

x =  zip(std_chain.params(),std_chain.mean(),std_chain.std())
for i in x:
    print '%s: \t %3.4f \pm %3.4f (no quadratic term)'%i


# In[58]:

std_chain.best_fit()['lnl']


# In[59]:

std_chain.best_fit()['params.H0']


# In[60]:

std_chain.std()


# In[61]:

hist(std_chain['lnl'], weights = std_chain['weight'], bins = 30);


# In[62]:

std_chain.params()


# In[63]:

std_chain.likegrid()


# In[ ]:




# In[ ]:



