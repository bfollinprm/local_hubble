
# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab')
get_ipython().magic(u'matplotlib inline')
rc('text', usetex = True)


# In[2]:

import pyfits as fits
from numpy import polyfit
from astropy.utils.data import download_file
import pandas as pd
from lmfit import Parameters, minimize, Minimizer


# In[3]:

### Download data


### Cepheid table
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


maser_distance = {'mu':7.2, 'e_mu':0.32} 
maser_distance = pd.DataFrame(data = maser_distance, index = arange(1))


# In[4]:

cepheids;


# In[5]:

Sne;


# In[8]:

class CosmoModel(object):
    def __init__(self, cepheids, Sne, ignore_field = None, IMrms_cut = None, flip_cut = False):
        
        self.cepheids = cepheids.copy()
        self.Sne = Sne.copy()
        self.fields = set(self.cepheids.Field)
        if IMrms_cut is None: IMrms_cut = [0, np.inf]
        if ignore_field is not None:

            ## Remove the fields in ignore_fields from the sample
            self.fields.discard(ignore_field)
        
        ####Cuts
        self.cepheids = self.cepheids.loc[self.cepheids.Flag != 'rej',:]
        self.cepheids = self.cepheids.loc[self.cepheids.Flag !='rej,low',:]
        self.cepheids = self.cepheids.loc[self.cepheids.Flag !='low',:]
        #self.cepheids = self.cepheids.loc[(5.297 < self.cepheids.Period) & (self.cepheids.Period < 203.3), :]
        if flip_cut is False:
            self.cepheids = self.cepheids.loc[self.cepheids.IMrms < IMrms_cut[1],:]
            self.cepheids = self.cepheids.loc[self.cepheids.IMrms > IMrms_cut[0],:]
        else:
            self.cepheids = self.cepheids[(self.cepheids.IMrms >= IMrms_cut[1]) | (self.cepheids.IMrms <= IMrms_cut[0])] 
            #self.cepheids = self.cepheids.loc[self.cepheids.IMrms <= IMrms_cut[0],:]

        if ignore_field is not None:
                self.cepheids = self.cepheids.loc[self.cepheids.Field != ignore_field,:]
                self.Sne = self.Sne.loc[self.Sne.Host != ignore_field, :]
        print 'kept %4.0f total cepheids'%self.cepheids.shape[0]

        self.params = Parameters()
        self.params.add('z_p_n4258', value = 25.5)
        self.params.add('b', value = -3, vary = True)
        self.params.add('Z', value = 0.25, vary = True)
        self.params.add('a_nu', value = 0.47)
        for field in self.fields:
                self.params.add('dmu_'+field, value = 0)
                self.params.add('R_'+field, value = 0, vary = False)
        #        self.params.add('ddmu_'+field, value = 0)
        #self.params['ddmu_n4258'].vary = False
        self.params['dmu_n4258'].vary = False
        self.params['dmu_n4258'].value = 0
        self.params.add('m04258', value = 30)
        self.params.add('m04258bias', value = 0, vary = False)
        self.params.add('mu_geometric', value = 7.6)
        self.params.add('R', value = 0.410, vary = False)
        self.params.add('R2', value = 0, vary = False)
        self.params.add('Q', value = 0, vary = False)
    def Fitted_Mag(self, cepheids, **params):
        
        cepheids.loc[:,'FittedMag'] = (params['z_p_n4258'] 
                        + params['b'] * log10(cepheids.Period) 
                        + params['Z'] * (cepheids.ObyH - mean(cepheids.ObyH))##8.9)#mean(cepheids.loc[cepheids.Field == 'n4258', 'ObyH'])) 
                        + params['R'] * cepheids.VtoI
                        + params['R2'] * cepheids.VtoI**2
                        + params['Q'] * cepheids.Bias
                       )
        for field in (self.fields):
            cepheids.loc[cepheids.Field == field,'FittedMag'] = (cepheids.loc[cepheids.Field == field, 'FittedMag'] 
                                                                 + params['dmu_'+field]
                                                                 + params['R_'+field] * cepheids.VtoI
                                                                )
        return cepheids
    
    def Fitted_Sne(self, Sne, **params):

        Sne.loc[:, 'fitted_abs_mag'] = params['m04258'] + params['m04258bias']
        for field in self.fields:
            Sne.loc[Sne.Host == field, 'fitted_abs_mag'] += params['dmu_'+field]

        return Sne
    
    def ceph_residual(self, cepheids, **params):
        cepheids.loc[:,:] = self.Fitted_Mag(cepheids, **params)
        result = (self.cepheids.F160Wmag-cepheids.FittedMag)/self.cepheids.e_F160Wmag
        return result
    
    def Sne_residual(self, Sne, **params):
        Sne.loc[:,:] = self.Fitted_Sne(Sne, **params)
        result = (self.Sne.m0_viPlus5a_v - 5 * 0.698 - self.Sne.fitted_abs_mag)/self.Sne['sigma']
        return result

    def priors(self, params):
        ### A_nu
        result = [(params['a_nu'] - 0.698)/0.00225]
        result+=[(params['mu_geometric'] - 7.6)/0.3]
        for field in (self.fields):
            result += [params['R_'+field]/0.2]
        #result+=[(params['b'] + 3.3)/0.1]
        #result+=[(params['Z'] + .21)/0.09]
        #result+=[(params['z_p_n4258'] - 26.36)/0.07]
        return array(result)
    
    def residual(self, params):
        Sne_result = self.Sne_residual(self.Sne, **params)
        ceph_result = self.ceph_residual(self.cepheids, **params)
        prior_result = self.priors(params)
        #weights = self.cepheids.IMrms / sum(self.cepheids.IMrms)
        result = concatenate((Sne_result, ceph_result, prior_result))
        #result = sum(Sne_result)
        #result += sum(ceph_result)# /weights)
        #result += sum(prior_result)
        return result
    
    def __call__(self):
        #out = minimize(self.residual, 
        #               self.params,
        #               scale_covar = False
        #              #method = 'cg'
        #               )
        mini = Minimizer(self.residual, self.params)
        out = mini.emcee(burn = 10000, steps  = 60000, thin = 1, workers = 1, params = self.params)
        self.H0 = 10**(out.params['a_nu'].value + 5 + 
                       0.2 * (out.params['m04258'].value - 5*log10(out.params['mu_geometric'].value) - 25))
        #print  5*log10(out.params['mu_geometric'].value) + 25
        self.e_H0 = model.H0 * sqrt((out.params['a_nu'].stderr * log(10))**2 
                            + (log(10)/5 *out.params['m04258'].stderr )**2
                            + (out.params['mu_geometric'].stderr/out.params['mu_geometric'].value)**2)
        return out


# In[9]:

model = CosmoModel(cepheids, Sne)
result = model()
print 'H_0 = %3.3f \pm %3.3f'%(model.H0, model.e_H0)
print '\chi^2 is ', result.chisqr, 'on around %4.0f degrees of freedom'%(model.cepheids.shape[0])
print 'reduced chisquare is %3.3f'%(result.chisqr/(model.cepheids.shape[0]+model.Sne.shape[0]+3 - 14))
for key in result.params.keys():
    if result.params[key].vary == True:
        print '%s = %3.3f \pm %3.3f'%(key, result.params[key].value, result.params[key].stderr)


# In[ ]:

percent_error = {}
for key in result.params.keys():
    if result.params[key].value != 0:
        percent_error[key] = result.params[key].stderr/result.params[key].value * 100
    else:
        percent_error[key] = 0
print percent_error['a_nu'], percent_error['m04258'], percent_error['mu_geometric']


# In[ ]:

Sne = Sne.sort('Host')
Sne.loc[:,'mu_4258'] = Sne.mu_0_Best.values - Sne.DeltaMu_0.values
true_dist = 5*log10(7.6) + 25
error = Sne.e_mu_0_Best
errorbar(arange(1,9), Sne.mu_4258-true_dist, yerr= error, fmt = 'o')
#errorbar(arange(1,9), values - SneMerged.ddmu, yerr= error, fmt = 'o')
#fill_between([0,9], 
#             [5*log10(result.params['mu_geometric'].value + result.params['mu_geometric'].stderr) + 25]*2, 
#            y2 = [5*log10(result.params['mu_geometric'].value - result.params['mu_geometric'].stderr) + 25]*2,
#             color = 'g',
#             alpha = 0.5
#            )
plot([0,9], [mean(Sne.mu_4258)-true_dist]*2, label = 'mean of data')
plot([0,9], [5*log10(7.2) + 25-true_dist]*2, label = '1999 value')
plot([0,9], [5*log10(7.6) + 25-true_dist]*2, label = '2013 value')
ylabel (r'$\mu_{4258}-{\mu}_{4258}^{\rm maser 2013}$', fontsize = 16)
xlim(0,9)
legend(loc = 4)
xticks(arange(1,9),Sne.Host.values);
Sne


# In[ ]:

print 'param \t \t value \t \t error \t \t % error'
for key in result.params.keys():
    if len(key)< 8: 
        disp_key = '%s\t'%key
    else:
        disp_key = key
    if result.params[key].vary != False:
        print '%s \t %f \t %f \t %f'%(disp_key, 
                                   result.params[key].value, 
                                   result.params[key].stderr, 
                                   abs(result.params[key].stderr/result.params[key].value)*100
                                  )

H_0_error = model.H0 * sqrt((result.params['a_nu'].stderr * log(10))**2 
                            + (log(10)/5 *result.params['m04258'].stderr )**2
                            + (result.params['mu_geometric'].stderr/result.params['mu_geometric'].value)**2)

H_0_frac_error = H_0_error/model.H0
print '%s \t %f \t %f \t %f'%('H_0      ',model.H0,H_0_error , H_0_frac_error)


# In[ ]:

fields = set(model.cepheids.Field)
for i, field in enumerate(fields):
    figure()
    x = log10(model.cepheids.loc[model.cepheids.Field == field, 'Period'])
    #result.params['b'].value * log10(model.cepheids.loc[model.cepheids.Field == field, 'Period'].values)
    y = (model.cepheids.loc[model.cepheids.Field ==field, 'F160Wmag'].values 
         -result.params['R'] * model.cepheids.loc[model.cepheids.Field == field, 'VtoI']
         -result.params['R_'+field] * model.cepheids.loc[model.cepheids.Field == field, 'VtoI']
         -result.params['Q'] * model.cepheids.loc[model.cepheids.Field == field, 'Bias']
         -(result.params['b'] * x + result.params['z_p_n4258'] + result.params['dmu_%s'%field])
        )
    yerr = model.cepheids.loc[model.cepheids.Field ==field, 'e_F160Wmag'].values
                                    
    errorbar(x , y , yerr= yerr, label = field, fmt = 'o')
    print sum((y/yerr)**2)/(x.size - 4), 1/sqrt(x.size - 4)
    legend(loc = 4)
    x = array([min(x), max(x)])
    plot(x, x * 0)
    #plot(x, result.params['b'] * x + result.params['z_p_n4258'] + result.params['dmu_%s'%field])
    gca().invert_yaxis()
    #xlim(0.4, 2.5)
    #ylim(20,30)
    
    


# In[ ]:

result.params


# In[ ]:

newcepheids = cepheids.sort_values(by = 'IMrms')
newcepheids = newcepheids.loc[newcepheids.Flag == '-',:]
newcepheids.index = arange(newcepheids.shape[0])
#newcepheids


# In[ ]:

indices = linspace(0, newcepheids.shape[0], 10).astype('int')

cuts = zip(roll(newcepheids.loc[indices, 'IMrms'].values, 1), newcepheids.loc[indices, 'IMrms'].values)[1:]
hvals = []
herr = []
cutlist = []
for cut in cuts:
    if not isfinite(cut[1]):
        cut = (cut[0], inf)
    if not isfinite(cut[0]):
        cut = (0, cut[1])
    #cut = (0, cut[1])
    print cut
    #print 'removing cephieds with IM_RMS %3.3f to %3.3f'%(cepheids.loc[i, 'IMrms'], cepheids.loc[i+1, 'IMrms'])
    #newcephs = cepheids.drop(cepheids.index[arange(i,i+1)])
    model = CosmoModel(newcepheids, Sne, IMrms_cut = array(cut), flip_cut = False)
    result = model()
    print 'H_0 = %3.2f \pm %3.2f'%(model.H0, model.e_H0)
    hvals.append(model.H0)
    herr.append(model.e_H0)
    cutlist.append(cut[0])
    for key in result.params.keys():
        if result.params[key].vary == True:
            if key[:4] != 'dmu_':
                print '%s = %3.2f \pm %3.2f'%(key, result.params[key].value, result.params[key].stderr)
    print '\n'


# In[ ]:

errorbar(arange(0.5, len(hvals)), hvals, yerr = herr, fmt = 'o')
model = CosmoModel(newcepheids, Sne)
result = model()
xvals = arange(-1, len(hvals)+1)
yvals = [model.H0] *xvals.size       
plot(xvals,yvals, color = 'g')
chi2val = (model.H0- array(hvals[:-1]))/array(herr[:-1])
#print 'chi2 is %3.2f'%chi2val
print sum(chi2val[1:]**2)
cutlist.append(r'$\infty$')
ylabel(r'$H_0 {\rm \, \,[km/s/Mpc]}$', fontsize = 18)
xlabel(r'${\rm Im}_{\rm rms} {\rm \, cut}$', fontsize = 16)
xticks(arange(len(hvals)+1),cutlist, rotation = 90)
xlim((0, len(hvals) ))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



