class Cepheid(object):
    def __init__(self, **kwargs):
        self.period = kwargs.pop('Period',0)
        self.mag = kwargs.pop('mag',0)
        self.color = kwargs.pop('Color',0)
        self.wmag = self.mag - 0.410 * self.color
        self.Z = kwargs.pop('Z',0)
        self.mag_error = kwargs.pop('e_mag',0)
        self.field = kwargs.pop('Field', '')
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __call__(self, m):
        return 0
    
names = ['Field', 'junk','Sn1a','Filters','mag_av', 'sigma', 'delta_mag','dmag_error','mu_0_Best','mu_best_error']    
class Sne(object):
    def __init__(self, **kwargs):
        self.field = kwargs.pop('Field', '')
        self.name = kwargs.pop('Sn1a', '')
        self.mag_av = kwargs.pop('mag_av',0)
        self.emag_av = kwargs.pop('sigma',0)
        self.delta_mag = kwargs.pop('delta_mag',0)
        self.edelta_mag = kwargs.pop('dmag_error',0)
        self.mu0_best = kwargs.pop('mu_0_Best',0)
        self.emu0_best = kwargs.pop('emu0_Best',0)
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __call__(self, m):
        return 0


class Cepheid_Collection(object):
    def __init__(self, cepheids, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.cepheids = cepheids
        self.period, self.mag, self.color, self.wmag, self.Z, self.mag_error, self.field = map(array, zip(*[[
                    x.period,
                    x.mag,
                    x.color,
                    x.wmag,
                    x.Z,
                    x.mag_error,
                    x.field
                                                            ] for x in cepheids]))
        self.period
        self.fields = ['n3370', 'n4536', 'n3982', 'n5584', 'n4639', 'n4038', 'n4258', 'n3021', 'n1309']
        self.Z = self.Z

    def plot(self, **kwargs):
        fig = kwargs.pop('fig', figure())
        ax = kwargs.pop('ax', subplot(111))
        field = kwargs.pop('field', self.fields[0])
        error = kwargs.pop('error', False)
        if error == False:
            xvals, yvals = zip(*[(x.period, x.wmag) for x in self.cepheids if x.field == field])
            ax.scatter(xvals, yvals , **kwargs);
            ax.invert_yaxis()
            ax.set_xscale('log')
        if error == True:
            xvals, yvals = zip(*[(x.period, x.wmag) for x in self.cepheids if x.field == field])
            ax.scatter(xvals, yvals , **kwargs);
            ax.invert_yaxis()
            ax.set_xscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins = 4))
        plt.close(fig)
        return ax
        
class Sne_Collection(object):
    def __init__(self, Sne, **kwargs):
        for key, value, in kwargs.items():
            setattr(self, key, value)
        self.fields = ['n3370', 'n4536', 'n3982', 'n5584', 'n4639', 'n4038', 'n4258', 'n3021', 'n1309']
        (self.field,self.name,self.mag_av,self.emag_av,
         self.delta_mag,self.edelta_mag,self.mu0_best,self.emu0_best) = map(array, zip(*[[
                    x.field,
                    x.name,
                    x.mag_av,
                    x.emag_av,
                    x.delta_mag,
                    x.edelta_mag,
                    x.mu0_best,
                    x.emu0_best
                                                            ] for x in Sne]))


from scipy.optimize import curve_fit
class PL_Relation():
    def __init__(self, cepheids, **kwargs):
        self.metallicity = kwargs.pop('metallicity', True)
        self.quad = kwargs.pop('quad', False)
        self.cubic = kwargs.pop('cubic', False)
        bestfit = kwargs.pop('fit', True)
        self.data = cepheids
        self.parnames = [r'%s PL intercept'%field for field in self.data.fields]
        self.parnames.append(r'$b$')
        if self.metallicity: self.parnames.append(r'$Z_p$')
        if self.quad: self.parnames.append(r'$b_2$')
        if self.cubic: self.parnames.append(r'$b_3$')
        
        num_params = len(self.data.fields) + 1 + self.metallicity + self.quad + self.cubic
        for key, value in kwargs.items():
            setattr(self, key, value)

        if bestfit:
            self.popt, self.pcov = curve_fit(self.fit, 
                                         self.data, 
                                         self.data.wmag,
                                         sigma = self.data.mag_error,
                                         p0 = ones(num_params))

            self.params = dict(zip(self.parnames,self.popt))
    def fit(self, data, *args, **kwargs):
        absolute_mag = kwargs.pop('absolute_mag',False)
        cntr = len(data.fields)
        mag = log10(data.period) * args[cntr]
        cntr += 1
        if self.metallicity:
            mag += log10(data.Z) * args[cntr]
            cntr += 1
        if self.quad:
            mag += log10(data.period)**2 * args[cntr]
            cntr += 1
        if self.cubic:
            mag += log10(data.period)**2 * args[cntr]
            cntr += 1
            
        if not absolute_mag:
            for i, field in enumerate(data.fields):
                mask = array([(el == field) for el in data.field])
                mag += mask * args[i]
        return mag
    
    def __call__(self, val, absolute_mag = False, params = None):
        if params == None: 
            try:
                params = self.popt
            except:
                print 'param values not supplied'
                return 0
        return self.fit(val, *params, absolute_mag = absolute_mag)
        