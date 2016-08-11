class Model(object):
    def __init__(self, cepheids, sne, params = None):
        '''
        params should be a dictionary of param: param_settings pairs, 
        where param_settings = [<start>, <vary (boolean)>]
        '''
        
        self.cepheids = cepheids
        self.sne = sne
        self.fields = set(self.cepheids.Field)
        
        
        ####Cuts
        self.cepheids = self.cepheids.loc[self.cepheids.Flag != 'rej',:]
        self.cepheids = self.cepheids.loc[self.cepheids.Flag !='rej,low',:]
        self.cepheids = self.cepheids.loc[self.cepheids.Flag !='low',:]
        #self.cepheids = self.cepheids.loc[(5.297 < self.cepheids.Period) & (self.cepheids.Period < 203.3), :]
        #self.cepheids = self.cepheids.loc[self.cepheids.IMrms < 5,:]
        #self.cepheids = self.cepheids.loc[self.cepheids.Field != 'n4536',:]
        #self.cepheids = self.cepheids.loc[self.cepheids.Field != 'n3021',:]

        self.params = Parameters()
        self.params.add('z_p_n4258', value = 25.5)
        self.params.add('b', value = -3, vary = True)
        self.params.add('Z', value = 0, vary = True)
        self.params.add('a_nu', value = 0.47)
        for field in self.fields:
                self.params.add('dmu_'+field, value = 0)
        self.params['dmu_n4258'].vary = False
        self.params['dmu_n4258'].value = 0
        self.params.add('m04258', value = 30)
        self.params.add('m04258bias', value = 0, vary = False)
        self.params.add('mu_geometric', value = 7.2)
        self.params.add('R', value = 2.50, vary = False)
        self.params.add('R2', value = 0, vary = False)
        self.params.add('Q', value = 0, vary = True)
        if params is not None:
            for key in params.keys():
                self.params.add(key, value = params[key][0], vary = params[key][1])

    def Fitted_Mag(self, cepheids, **params):
        
        cepheids.loc[:,'FittedMag'] = (params['z_p_n4258'] 
                        + params['b'] * log10(cepheids.Period) 
                        + params['Z'] * (cepheids.ObyH - mean(cepheids.loc[cepheids.Field == 'n4258', 'ObyH'])) 
                        + params['R'] * cepheids.VtoI
                        + params['R2'] * cepheids.VtoI**2
                        + params['Q'] * cepheids.Bias
                       )
        for field in (self.fields):
            cepheids.loc[cepheids.Field == field,'FittedMag'] = (cepheids.loc[cepheids.Field == field, 'FittedMag'] 
                                                                 + params['dmu_'+field])
        return cepheids
    
    def Fitted_sne(self, sne, **params):

        sne.loc[:, 'fitted_abs_mag'] = params['m04258'] + params['m04258bias']
        for field in self.fields:
            sne.loc[sne.Host == field, 'fitted_abs_mag'] += params['dmu_'+field]
        return sne
    
    def ceph_residual(self, cepheids, **params):
        cepheids.loc[:,:] = self.Fitted_Mag(cepheids, **params)
        result = (self.cepheids.F160Wmag-cepheids.FittedMag)/self.cepheids.e_F160Wmag
        return result
    
    def sne_residual(self, sne, **params):
        sne.loc[:,:] = self.Fitted_sne(sne, **params)
        result = (self.sne.m0_viPlus5a_v - 5 * 0.698 - self.sne.fitted_abs_mag)/self.sne['sigma']
        return result

    def priors(self, params):
        ### A_nu
        result = [(params['a_nu'] - 0.698)/0.00225]
        result+=[(params['mu_geometric'] - 7.2)/0.32]
        return array(result)
    
    def residual(self, params):
        sne_result = self.sne_residual(self.sne, **params)
        ceph_result = self.ceph_residual(self.cepheids, **params)
        prior_result = self.priors(params)
        result = concatenate((sne_result, ceph_result, prior_result))
        return result
    
    def __call__(self):
        out = minimize(self.residual, 
                       self.params
                      #method = 'cg')
                       )
        self.H0 = 10**(out.params['a_nu'].value + 5 + 
                       0.2 * (out.params['m04258'].value - 5*log10(out.params['mu_geometric'].value) - 25))
        try:
            self.e_H0 = model.H0 * sqrt((out.params['a_nu'].stderr * log(10))**2 
                            + (log(10)/5 *out.params['m04258'].stderr )**2
                            + (out.params['mu_geometric'].stderr/out.params['mu_geometric'].value)**2)
        except:
            self.e_H0 = None


        return self.H0, self.e_H0, out
