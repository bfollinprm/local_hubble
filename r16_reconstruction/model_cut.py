import cosmoslik as cs
from numpy import *
import pandas as pd
from sklearn.cluster import KMeans
param = cs.param_shortcut('start','scale')
class Model(cs.SlikPlugin):
    
    def __init__(self,
                 varyR = False,
                 second_order = False,
                 correlated_errs = False,
                 rprior = 0.0,
                 color_cutoff = False,
                 filename = 'test.chain',
                 proposal = 'proposal.covmat'
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
        self.second_order = second_order
        self.correlated_errs = correlated_errs
        self.rprior = rprior
        self.color_cutoff = color_cutoff
        ### data    
        self.cepheids, self.sne = self.__get_data__()
        self.fields = self.cepheids.groupby('Field')

        #### Cluster the data
        classifier = KMeans(n_clusters = 2, n_init = 1000)
        cephs = self.cepheids.copy()
        features = ['VminusI', 'Period', 'Z']
        for feature in features:
            if feature =='Period':
                cephs[feature] = log10(self.cepheids.Period)/(log10(self.cepheids.Period).max() - log10(self.cepheids.Period).min())
            else:
                cephs[feature] = self.cepheids[feature]/((self.cepheids[feature]).max() - (self.cepheids[feature]).min())
        classifier.fit(cephs[features])
        classification = classifier.predict(cephs[features])



        self.cepheids['classes'] = classification
        if mean(self.cepheids.loc[self.cepheids.classes == 1, 'Period']) > mean(self.cepheids.loc[self.cepheids.classes == 0, 'Period']):
            self.cepheids.classes = -self.cepheids.classes + 1

        self.cepheids = self.cepheids[self.cepheids.classes == 0]
        print mean(self.cepheids.Period)
        
        starts = {'m101': 29.143079353587371,  'n3370': 32.129744303628165, 
                  'n3447': 31.932213325716038, 'n1448': 31.360344037644502, 
                  'u9391': 32.930679386412685, 'n3982': 31.733859058685638, 
                  'n1309': 32.557100354527115, 'n1365': 31.250424930409952, 
                  'lmc': 18.473189310256739,   'm31': 24.458640539140053, 
                  'n3021': 32.492603834744138, 'n4424': 30.705218105160416, 
                  'n4258': 29.359512363837116, 'n5917': 32.188502027063343, 
                  'n5584': 31.831167288091724, 'n2442': 31.564541453741874, 
                  'n7250': 31.484075847498222, 'n4038': 31.267464703999412, 
                  'n1015': 32.658376618425386, 'n4536': 30.880118695205464, 
                  'n3972': 31.594769594460779, 'n4639': 31.575107430217763
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
        self.M_ceph = param(-5.6765476611409191, 0.1)
        self.b = param(-3.1748406440092007, 0.02)
        self.gamma = param( -0.15278307999698593, 0.05)
        self.R = 0.39
        
        if self.color_cutoff:
            self.E_max = param(0.56366698226982348, 0.2, min = 0.4, max = 1.8)
        if self.second_order:
            self.b2 = param(0, 0.01)
            self.bz = param(0,0.01)
            self.bR = param(0,0.01)
            self.zR = param(0,0.01)     
        if self.correlated_errs:
            self.k_lmc = param(0.05, 0.01, min = 0, max = 1)
            self.k_m31 = param(0.05, 0.01, min = 0, max = 1)
            
        
        
        #E(V-I)
        self.C0 = param(0.75, 0.01)
        self.CP_l = param(0.21708628141955091, 0.01)
        self.CP_h = param(0.21708628141955091, 0.01)
        self.CZ = param(-0.022939433819726411, 0.05)
        
        #SNe
        self.a_B = cs.param(start= 0.717, scale = 0.00176, gaussian_prior = (0.71629365006647927, 0.00176))
        self.M_sne = param(-19.287702489694055, 0.1)
        
        #nuisance parameters
        self.dz = cs.param(start = 0.0028778905772363084, scale = 0.01, gaussian_prior = (0, 0.03))

        if proposal == 'None':

          self.sampler = cs.get_plugin('samplers.metropolis_hastings')(self,
                                                       num_samples = 1.0e8,
                                                       proposal_scale = 0.2,
                                                       output_file = filename,
                                                       output_extra_params = ['H0'],
                                                       print_level = 1,
                                                      )

        else: 
          self.sampler = cs.get_plugin('samplers.metropolis_hastings')(self,
                                                       num_samples = 1.0e8,
                                                       proposal_scale = 1.6,
                                                       output_file = filename,
                                                       output_extra_params = ['H0'],
                                                       print_level = 1,
                                                       proposal_cov = proposal
                                                      )
        
        self.priors = cs.get_plugin('likelihoods.priors')(self)

    def __call__(self):
        self.H0 = 10**(0.2 * self.M_sne +  self.a_B + 5)
        return self.__cost__()
    
    def __cost__(self):
        '''
        Calculates the residuals (f_i(\theta) - y_i)/sigma_i.
        Returns:
            resid: numpy array of (f_i(\theta) - y_i)/sigma_i
        '''
        self.cepheids['Extinction'] = (self.cepheids.VminusI 
                                    - self.C0
                                    - (log10(self.cepheids.Period)-1) * (self.CP_h * (self.cepheids.Period >= 10) 
                                                                        + self.CP_l * (self.cepheids.Period < 10)
                                                                        )
                                    - self.CZ * (self.cepheids.Z - 8.9)
                                          )
        if self.color_cutoff:
            
            cepheids = self.cepheids[self.cepheids.Extinction < self.E_max]
            cepheids.reset_index(drop = True, inplace = True)
        else:
            cepheids = self.cepheids

        mu_cephs = zeros(cepheids.m_H.size)
        mu_sne = zeros(self.sne.Host.size)
        R_field = ones(cepheids.m_H.size) *0.39
        for field in self.fields.groups.keys():
            mu_cephs += array(cepheids.Field == field) * self['mu_'+field]
            mu_sne += array(self.sne.Host == field) * self['mu_'+field]
            if self.varyR:
                R_field += array(cepheids.Field == field) * self['R_'+field]




        if self.color_cutoff:
            determinant = sum(log(cepheids.err**2+ 0.08**2*R_field**2)) + (cepheids.err.size)*log(2*pi)
        else:
            determinant = sum(log(cepheids.err**2+ 0.08**2*R_field**2)) + (cepheids.err.size)*log(2*pi)
            



            

    
        yi = cepheids.m_H
        fi = (self.M_ceph 
               + self.gamma * (cepheids.Z-8.9)
               + (log10(cepheids.Period) - 1) * self.b
               + R_field * cepheids.Extinction
               + mu_cephs
               + self.dz * (array(cepheids.Field == 'galaxy')
                            + array(cepheids.Field == 'lmc')
                           )

               + array(cepheids.Field == 'galaxy') * (5 * log10(1.0e-3/cepheids.parallax) 
                                                           + 25 
                                                           - cepheids.LK
                                                          )
               )
        if self.second_order:
            fi += ((log10(cepheids.Period)-1)**2 * self.b2
                   +(log10(cepheids.Period) -1)*(cepheids.Z - 8.9)*self.bz
                   +(log10(cepheids.Period) -1) * cepheids.Extinction * self.bR
                   +(cepheids.Z - 8.9) * cepheids.Extinction * self.zR
                   )
            
        if self.correlated_errs:
            cov = pd.DataFrame(diag(cepheids.err**2 + 0.08**2*R_field**2), 
                               index = cepheids.Field,
                               columns = cepheids.Field
                               )
            lmc = array(cov['lmc','lmc'])
            n = lmc.shape[0]
            for i in arange(n-1):
                for j in arange(i+1, n-1):
                    lmc[i,j] = self.k_lmc * sqrt(lmc[i,i] *lmc[j,j])
                    lmc[j,i] = lmc[i,j]
            eigs, vecs = eigh(lmc)
            eigs *= exp((sum(log(cepheids.err**2)) - sum(log(eigs)))/n)
            cov['lmc','lmc'] = dot(vecs.T, dot(diag(eigs), vecs))
            
            m31 = array(cov['m31','m31'])
            n = m31.shape[0]
            for i in arange(n-1):
                for j in arange(i+1, n-1):
                    m31[i,j] = self.k_m31 * sqrt(m31[i,i] *m31[j,j])
                    m31[j,i] = m31[i,j]
            eigs, vecs = eigh(m31)
            eigs *= exp((sum(log(cepheids.err**2)) - sum(log(eigs)))/n)
            cov['m31','m31'] = dot(vecs.T, dot(diag(eigs), vecs))
            covinv = inv(cov)
            determinant = sum(log(eigs)) + (eigs.size)*log(2*pi)
        else:
            covinv = diag(1/(cepheids.err**2 + 0.08**2*R_field**2))
            
        res = yi - fi
        cephs = dot(res.T, dot(covinv, res))
        
        ### Sne
        yi = self.sne['m^B_0']
        fi = self.M_sne + mu_sne
        covinv = diag(1.0/self.sne.err**2)
        res = yi-fi
        sne = dot(res.T, dot(covinv, res))
        
        ### Intrinsic color from 10.1051/0004-6361:20030354 and 10.1051/0004-6361:20040222
        #lmc data
        lmc = self.fields.get_group('lmc')
        # yi = (log10(lmc.Period) * (0.316 * (lmc.Period > 10) 
        #                            +0.160 * (lmc.Period <= 10)
        #                           )
        #       +(0.380 * (lmc.Period > 10) 
        #         +0.501 * (lmc.Period <= 10)
        #        )
        #      )
        yi = (log10(lmc.Period) * (0.160 * (lmc.Period < 10) + (0.315 * (lmc.Period >= 10))) 
                               + (0.501 * (lmc.Period < 10) + (0.380 *(lmc.Period >= 10)))
                                )
        fi = (self.C0 + 
              (log10(lmc.Period) - 1) * (self.CP_l * (lmc.Period < 10) + self.CP_h * (lmc.Period >= 10)) 
              + self.CZ * (lmc.Z - 8.9)
               )
        covinv = diag(1.0/(zeros(lmc.Period.size) + 0.08**2))
        res = yi - fi

        lmc = dot(res.T, dot(covinv, res))
        
        galaxy = self.fields.get_group('galaxy')
        yi = 0.256 * log10(galaxy.Period) +0.497
        fi = (self.C0 + 
              (log10(galaxy.Period) - 1) * (self.CP_l * (galaxy.Period < 10) + self.CP_h * (galaxy.Period >= 10)) 
              + self.CZ * (galaxy.Z - 8.9)
               )
        res = yi - fi
        covinv = diag(1.0/(zeros(galaxy.Period.size) + 0.08 ** 2))
        galaxy = dot(res.T, dot(covinv, res))

        
        lnl = 0.5*(cephs+sne+lmc+galaxy+2*self.priors(self) + determinant)
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
