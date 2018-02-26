import cosmoslik as cs
from numpy import *
from numpy.random import rand
import pickle
import pandas as pd

param = cs.param_shortcut('start','scale')
class Model(cs.SlikPlugin):
    
    def __init__(self,
                 varyR = False,
                 multiple_pops = False,
                 rprior = 0.0,
                 color_cutoff = False,
                 filename = 'test.chain',
                 proposal = 'proposal.covmat',
                 dz_prior = 0.03,
                 color_correction = True,
                 only_pop = 0
                ):
        '''
        
        varyR: varies the reddening correction to E(V-I) applied to extra-galactic cepheids by host
        second_order: includes second-order corrections to the Cepheid PL relationship
        correlated_errs: allows for the possibility of the intrinsic width of the instability strip to give
                         correlated errors.
        rprior: set a prior on the extra-galactic Cepheid reddening laws (useful to prevent overfitting)
        color_cutoff: between [0,1). Cuts a percentage of Cepheids with high E(V-I) out of the sample.
        filename: file to save chain as.
        Bound methods:
        '''
        super(cs.SlikPlugin,self).__init__()
        self.varyR = varyR
        self.rprior = rprior
        self.multiple_pops = multiple_pops
        self.color_cutoff = color_cutoff
        ### data    
        self.cepheids, self.sne = self.__get_data__()
        self.cepheids.loc[:,'logP'] = log10(self.cepheids.Period)
        self.dz_prior = dz_prior
        self.color_correction = color_correction
        self.only_pop = only_pop
        print 'creating population probabilities'

        if multiple_pops:
            self.n_classes = 6
            #from sklearn.mixture import GaussianMixture
            #model = GaussianMixture(n_components = self.n_classes, max_iter = 1000, n_init = 1000, init_params= 'random')
            readfile = open('gaussian_mixture_model', 'r')
            model = pickle.load(readfile)
            readfile.close()
            features = self.cepheids[['logP', 'VminusI']]
            self.probs = model.predict_proba(features)
            names = ['Prob_' + string for string in (arange(self.n_classes)).astype('str')]
            for i,name in enumerate(names):
                self.cepheids[name] = self.probs[:,i]
                print 'effective number of cepheids in class', i, 'is ', sum(self.cepheids[name])

        if only_pop != 0:
            for i, name in enumerate(names):
              if i != only_pop - 1:
                self.cepheids[name] = 0
              else:
                self.cepheids[name] = self.probs[:, only_pop - 1]
                print name, 'reweighted'
                print 'effective number of cepheids in class', i, 'is ', sum(self.cepheids[name])

            #print self.cepheids.head(5)



        else:
            self.n_classes = 1
            self.cepheids['Prob_0'] = 1
            self.probs = ones([self.cepheids.m_H.size, 1])
            #self.cepheids['Prob_1'] = 0.5
            #self.multiple_pops = True


        self.fields = self.cepheids.groupby('Field')
        print 'initializing MCMC'
        
        starts = {'m101': 29.144437436768779,  'n3370': 32.085326573203133, 
                  'n3447': 31.925511937144158, 'n1448': 31.331612808727602, 
                  'u9391': 32.876957912796556, 'n3982': 31.691652487648998, 
                  'n1309': 32.538213812056632, 'n1365': 31.29349520994554, 
                  'lmc': 18.491674020476449,   'm31': 24.432759814673339, 
                  'n3021': 32.434801375871849, 'n4424': 30.913725676693797, 
                  'n4258': 29.352312664474258, 'n5917': 32.193993084145056, 
                  'n5584': 31.791532724280536, 'n2442': 31.535445988717694, 
                  'n7250': 31.486608822858241, 'n4038': 31.419939642742925, 
                  'n1015': 32.567894400797492, 'n4536': 30.920497941433734, 
                  'n3972': 31.65839514301733, 'n4639': 31.528312982189139
                  }
        
        ### Params
        for field in self.fields.groups.keys():
            if field == 'galaxy':
                self['mu_'+field] = 0
                if self.varyR:
                    self['R_'+field] = 0
            
            elif field == 'lmc':
                self['mu_'+field] = param(starts[field],0.04, gaussian_prior = (18.494, 0.0452))
                if self.varyR:
                    if self.rprior != 0:
                        self['R_'+field] = param(0, 0.05, gaussian_prior = (0, self.rprior), min = -0.39)
                    else:
                        self['R_'+field] = param(0, 0.05, min = -0.39)
                        
                        
            elif field == 'n4258':
                self['mu_'+field] = param(starts[field],0.05, gaussian_prior = (29.387, 0.02616*2.17))
                if self.varyR:
                    if self.rprior != 0:
                        self['R_'+field] = param(0, 0.05, gaussian_prior = (0, self.rprior), min = -0.39)
                    else:
                        self['R_'+field] = param(0, 0.05, min = -0.39)

            else:
                self['mu_'+field] = param(starts[field],0.1)
                if self.varyR:
                    if self.rprior != 0:
                        self['R_'+field] = param(0, 0.05, gaussian_prior = (0, self.rprior), min = -0.39)
                    else:
                        self['R_'+field] = param(0, 0.05, min = -0.39)


        
        #PL relationship
        # self['M_ceph_mean'] = param(-6, 0.6)
        # self['M_ceph_scale'] = 0.6
        # self['b_mean'] = param(-3, 0.3)
        # self['b_scale'] = 0.3
        # self['gamma_mean'] = param(0, 0.1)
        # self['gamma_scale'] = 0.1
        # self['C0_mean'] = param(0.75, 0.1)
        # self['C0_scale'] = 0.1
        # self['CP_mean'] = param(0.22, 0.1)
        # self['CP_scale'] = 0.1
        # self['CZ_mean'] = param(0, 0.1)
        # self['CZ_scale'] = 0.1
        for pop in arange(self.n_classes).astype('str'):
            if pop == '0':
                self['M_ceph' + pop] = param(-5.6335264585145186, 0.6)#, gaussian_prior = (self.M_ceph_mean, self.M_ceph_scale))
                self['b' + pop] = param(-3.0869163562829098, 0.3)#,gaussian_prior = (self.b_mean, self.b_scale))
                self['gamma'+pop] = param(-0.30093952151120151, 0.1)#,gaussian_prior = (self.gamma_mean, self.gamma_scale))
                self['R'+pop] = 0.39
                ## intrinsic color parameters
                if self.color_correction:
                    self['C0'+pop] = param(0.81728892456760971, 0.1,gaussian_prior = (0,10))
                    self['CP'+pop] = param(0.43267560914644093, 0.1, gaussian_prior = (0,10))#,gaussian_prior = (self.CP_mean, self.CP_scale))
                    self['CZ'+pop] = param(-0.38724106392553137, 0.1, gaussian_prior = (0,10))#, gaussian_prior = (self.CZ_mean, self.CZ_scale))
            else:
                if self.only_pop != 0:
                  self['M_ceph' + pop] = 0#, gaussian_prior = (self.M_ceph_mean, self.M_ceph_scale))
                  self['b' + pop] = 0#,gaussian_prior = (self.b_mean, self.b_scale))
                  self['gamma'+pop] = 0#,gaussian_prior = (self.gamma_mean, self.gamma_scale))
                  self['R'+pop] = 0
                  if self.color_correction:
                    self['C0'+pop] = 0#,gaussian_prior = (self.C0_mean, self.C0_scale))
                    self['CP'+pop] = 0#,gaussian_prior = (self.CP_mean, self.CP_scale))
                    self['CZ'+pop] = 0
                else:
                  self['M_ceph' + pop] = param(0, 0.01)#, gaussian_prior = (self.M_ceph_mean, self.M_ceph_scale))
                  self['b' + pop] = param(0, 0.01)#,gaussian_prior = (self.b_mean, self.b_scale))
                  self['gamma'+pop] = param(0, 0.01)#,gaussian_prior = (self.gamma_mean, self.gamma_scale))
                  self['R'+pop] = 0.39
                  if self.color_correction:
                    self['C0'+pop] = param(0, 0.01, gaussian_prior = (0, 0.3))#,gaussian_prior = (self.C0_mean, self.C0_scale))
                    self['CP'+pop] = param(0, 0.01, gaussian_prior = (0, 0.3))#,gaussian_prior = (self.CP_mean, self.CP_scale))
                    self['CZ'+pop] = param(0, 0.01, gaussian_prior = (0,0.3))
                #self['R'+pop] = param(0.39, 0.04, gaussian_prior = (0.39, 0.039))
                ## intrinsic color parameters

        
        if self.color_cutoff:
            self.E_max = param(0.56366698226982348, 0.2, min = 0.4, max = 1.8)
   
            
        
        

        
        #SNe
        self.a_B = cs.param(start= 0.717, scale = 0.00176, gaussian_prior = (0.71629365006647927, 0.00176))
        self.M_sne = param(-19.266417457292171, 0.1)
        
        #nuisance parameters
        self.dz = cs.param(start = -0.0095927164966368543, scale = 0.01, gaussian_prior = (0, self.dz_prior))

        if proposal == 'None':

          self.sampler = cs.get_plugin('samplers.metropolis_hastings')(self,
                                                       num_samples = 100000000,
                                                       proposal_scale = 0.2,
                                                       output_file = filename,
                                                       output_extra_params = ['H0'],
                                                       print_level = 1,
                                                      )

        else: 
          self.sampler = cs.get_plugin('samplers.metropolis_hastings')(self,
                                                       num_samples = 100000000,
                                                       proposal_scale = 2.5,
                                                       output_file = filename,
                                                       output_extra_params = ['H0'],
                                                       print_level = 1,
                                                       proposal_cov = proposal
                                                      )
        
        self.priors = cs.get_plugin('likelihoods.priors')(self)




        self.cepheids['Extinction'] = 0

    def __call__(self):
        self.H0 = 10**(0.2 * self.M_sne +  self.a_B + 5)
        return self.__cost__()
    
    def __cost__(self):
        '''
        Calculates the residuals (f_i(\theta) - y_i)/sigma_i.
        Returns:
            resid: numpy array of (f_i(\theta) - y_i)/sigma_i
        '''
        #### Assign classes to cepheids
        cepheids = self.cepheids.copy()
        cepheids.loc[:,'Int_Color'] =  zeros(cepheids.m_H.size)
        cepheids.loc[:,'m_theory'] =  zeros(cepheids.m_H.size)
        cepheids.loc[:,'R_field'] =  zeros(cepheids.m_H.size)
        cepheids.loc[:,'mu_cephs'] = zeros(cepheids.m_H.size)
        mu_sne = zeros(self.sne.Host.size)
        for field in self.fields.groups.keys():
            cepheids.loc[:,'mu_cephs'] += array(cepheids.Field == field) * self['mu_'+field]
            mu_sne += array(self.sne.Host == field) * self['mu_'+field]



        lnl = self.priors(self)

        class_num = self.n_classes

        cepheids.loc[:,'R_field'] = zeros(cepheids.m_H.size)
        if self.varyR:
	        for field in self.fields.groups.keys():
	            cepheids['R_field'] += array(cepheids.Field == field)*self['R_'+field]


        ### Cepheid likelihood
        res = zeros(cepheids.m_H.size)
        for cepheid_class in arange(class_num):
            pop = str(cepheid_class)
            prob = cepheids['Prob_'+pop]
            if (cepheid_class == 0) & (self.color_correction):
                int_color = (self['C0'+pop] 
                                        + (cepheids.logP - 1) * self['CP' + pop] 
                                        + self['CZ' + pop] * (cepheids.Z - 8.9)
                                      )
            elif (cepheid_class != 0) & (self.color_correction):
                int_color = (self['C0'+pop] + self['C00'] 
                                        + (cepheids.logP - 1) * (self['CP' + pop]  + self['CP0'])
                                        + (self['CZ' + pop] + self['CZ0']) * (cepheids.Z - 8.9)
                                      )
            else:
                int_color = 0
            if self.only_pop == 0:
                cepheids.loc[:,'Int_Color'] += int_color * prob
            elif cepheid_class == self.only_pop - 1:
                cepheids.loc[:,'Int_Color'] = int_color 
            else:
                pass


            Rvals  = cepheids.loc[:,'R_field'] + self['R'+pop]
            yi = ( cepheids.m_H- Rvals * (cepheids['VminusI'] - int_color) )
            fi = (self['M_ceph'+pop]
                  + self['gamma'+pop] * (cepheids.Z -8.9)
                  + self['b'+pop] * (cepheids.logP - 1)
                  + cepheids['mu_cephs']
                  + self['dz'] * (array(cepheids.Field == 'galaxy') + array(cepheids.Field == 'lmc'))
                  + array(cepheids.Field == 'galaxy') * (5 * log10(1.0e-3/cepheids.parallax) + 25 - cepheids.LK)
                  )
            if (cepheid_class != 0):
                pop = '0'
                fi += (self['M_ceph'+pop]
                  + self['gamma'+pop] * (cepheids.Z -8.9)
                  + self['b'+pop] * (cepheids.logP - 1)
                  )
            #determinant = sum(log(cephs.err**2+ 0.08**2*cephs.R_field**2*cephs['Prob_'+pop]**2))
            res += (yi - fi) * prob
        covinv = diag(1/(cepheids.err**2 + 0.08**2*cepheids.R_field**2))

        chisquared = (dot(res.T, dot(covinv, res)))/2.0
        lnl += chisquared 

        ### Sne likelihood
        yi = self.sne['m^B_0']
        fi = self.M_sne + mu_sne
        covinv = diag(1.0/self.sne.err**2)
        res = yi-fi
        lnl += dot(res.T, dot(covinv, res))/2.0
        
        ### Intrinsic color from 10.1051/0004-6361:20030354 and 10.1051/0004-6361:20040222
        #lmc data
        if self.color_correction:
            lmc = cepheids.groupby('Field').get_group('lmc')
            yi = (log10(lmc.Period - 1) * (0.160 * (lmc.Period < 10) + (0.315 * (lmc.Period >= 10))) 
                                   + (0.661 * (lmc.Period < 10) + (0.695 *(lmc.Period >= 10)))
                                    )
            fi = lmc['Int_Color']
            covinv = diag(1.0/(zeros(lmc.Period.size) + 0.08**2))
            res = yi - fi
            lnl += dot(res.T, dot(covinv, res))/2.0
            
            galaxy = cepheids.groupby('Field').get_group('galaxy')
            yi = 0.256 * log10(galaxy.Period) +0.497
            fi = galaxy['Int_Color']
            res = yi - fi
            covinv = diag(1.0/(zeros(galaxy.Period.size) + 0.08 ** 2))
            lnl += dot(res.T, dot(covinv, res))/2.0
        

        #print cepheids[:5]['Int_Color']
        #self.cepheids = cepheids
        return lnl
        
        
    def __get_data__(self):
        '''
        Grabs the cepheids and sne from the R16 sample
        returns: 
            cepheids: Cepheid dataframe
            sne: sne dataframe
        '''
        filename = '../data/r16_table4.out'
        sne_start = 40
        sne_end = 59
        sne_lines = arange(sne_start,sne_end)
        sne = pd.DataFrame(columns = ['Host', 'sne', 'm^B_0', 'err'], index = arange(sne_end - sne_start))
        ceph_start = 70
        ceph_end = 2346
        cepheid_lines = arange(ceph_start,ceph_end)
        cepheids = pd.DataFrame(columns = ['Field','RA','DEC','ID','Period','VminusI','m_H','sigma_tot','Z'], 
                                index = arange(ceph_end - ceph_start),
                               dtype = 'float')
        f = file(filename)
        for i, line in enumerate(f):
            if i in sne_lines:
                sne.loc[i-sne_start] = line.lower().split()
            if i in cepheid_lines:
                cepheids.loc[i-ceph_start] = line.lower().split()

        f.close()
        cepheids = cepheids.apply(lambda x: pd.to_numeric(x, errors='ignore') );

        sne = sne.apply(lambda x: pd.to_numeric(x, errors='ignore') );
        
        
        parallaxes = {'bgcru': (2.23, 0.30,-0.15), 
                      'dtcyg':(2.19,0.33, -0.18), 
                      'ffaql':(2.64,0.16, -0.03),
                      'rtaur':(2.31, 0.19,-0.06),
                      'sscma':(0.348, 0.038, -0.04),
                      'sucas':(2.57,  0.33, -0.13 ),
                      'syaur':(0.428, 0.054, -0.04),
                      'tvul':(2.06,0.22,-0.09 ),
                      'wsgr':(2.30, 0.19, -0.06),
                      'xsgr':(3.17, 0.14, -0.02),
                      'ysgr':(2.13, 0.29, -0.15),
                      'betador':(3.26, 0.14, -0.02),
                      'delceph':(3.71,0.12,-0.01),
                      'etagem':(2.74,0.12,-0.02),
                      'lcar':(2.03,0.16,-0.05)
                     }
        parallaxes = pd.DataFrame.from_dict(parallaxes, orient = 'index', )
        parallaxes.reset_index(inplace=True)
        parallaxes.columns = ['ID', 'parallax', 'p_err', 'LK']
        cepheids = cepheids.merge(parallaxes, on = 'ID', how = 'left')
        

        cepheids.fillna({'parallax':1e-03, 'p_err':0, 'LK':0}, inplace = True);
        cepheids['err'] = sqrt(cepheids.sigma_tot**2 + (cepheids.p_err / cepheids.parallax * 5/log(10))**2)
        return cepheids, sne