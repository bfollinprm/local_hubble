import cosmoslik as cs  
import numpy as np 
import pandas as pd
from numpy.linalg import inv
from scipy.spatial import KDTree
import sys
from repoze.lru import lru_cache	
sys.setrecursionlimit(10000)
param = cs.param_shortcut('start', 'scale')



class Model(cs.SlikPlugin):
	def __init__(self, num_samples = 100000):
		super(Model,self).__init__()

		

		self.priors = cs.get_plugin('likelihoods.priors')(self)

		### Data
		self.cephs, self.sne = get_data()
		self.cephs = preprocess_cepheids(self.cephs)

		### Connector
		# Defines how to go from host to anchor galaxies
		self.get_cepheid_magnitude = ClosestNeighborMagnitude(self.cephs)


		self.params = cs.SlikDict()
		self.derived = cs.SlikDict()

		self.params.H0 = param(72, 1.5)
		self.params.a_B = cs.param(start= 0.717, scale = 0.00176, gaussian_prior = (0.71629365006647927, 0.00176))
		self.params.dz = param(0, 0.002, gaussian_prior = (0, 0.03))
		starts = {
			'm101': 29.144437436768779,  'n3370': 32.085326573203133, 
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
		distances = dict([('mu_'+key, param(starts[key], 0.05)) for key in starts.keys()])
		for key in distances.keys():
			self.params[key] = distances[key]


		### Priors
		self.params.mu_lmc = param(starts['lmc'],0.04, gaussian_prior = (18.494, 0.0452))
		self.params.mu_n4258 = param(starts['n4258'],0.05, gaussian_prior = (29.387, 0.02616*2.17))
		self.params.mu_galaxy = 0


		self.sampler = cs.get_plugin('samplers.metropolis_hastings')(
		self,
		num_samples = num_samples,
		proposal_scale = 0.5,
		output_file = 'test.chain',
		output_extra_params = ['derived.Msne'],
		print_level = 1,
		)


	def __call__(self):
		lnl = np.log1p(self.priors(self))
		#print 'lnl = {}'.format(lnl)
		self.derived.Msne = (np.log10(self.params.H0) - 5 - self.params.a_B)* 5

		### SNe
		fi = self.derived.Msne
		yi = self.sne[['Field','m_0B']].copy()
		grouped = yi.groupby('Field')
		ret = []
		for name, group in grouped:
			mu = self.params['mu_'+name]
			group['m_0B']= group['m_0B'] - mu
			ret += [group['m_0B']]
		yi = pd.concat(ret)
		lnl += 0.5 * (sum((fi - yi)**2/self.sne['err']**2))
		#print 'lnl = {}'.format(lnl)

		### Cepheids
		fi, mag_err = self.get_cepheid_magnitude(self.params)
		yi = self.cephs[['Field','m_H']].copy()
		grouped = yi.groupby('Field')
		ret = []
		for name, group in grouped:
			mu = self.params['mu_'+name]
			group['m_H']= group['m_H'] - mu
			ret += [group['m_H']]
		yi = pd.concat(ret)
		keep = fi.notnull()
		lnl += 0.5 * (sum((yi - fi - 0.39 * self.cephs['VminusI']).loc[keep]**2/mag_err.loc[keep]**2))


		#print 'lnl = {}'.format(lnl)
		return lnl



class ClosestNeighborMagnitude(object):
	from scipy.spatial import KDTree
	def __init__(self, cepheids, max_dist = 0.5):
		print 'building tree'
		cepheids['logP'] = cepheids.Period.apply(np.log10)
		cepheids['eM'] = 0
		self.cov = (cepheids[['logP', 'VminusI']]).cov()
		d, V = np.linalg.eigh(self.cov)
		d = np.diag(1/np.sqrt(d))
		W = np.dot(np.dot(V,d), V.T)
		transformed = np.dot(cepheids[['logP', 'VminusI']].values, W)
		cepheids['T1'] = transformed[:,0]
		cepheids['T2'] = transformed[:,1]
		self.cepheids = cepheids
		self.known = self.cepheids.loc[self.cepheids.Field.isin(['lmc','galaxy', 'n4258'])]
		self.tree = KDTree(self.known[['T1', 'T2']], leafsize=6)
		self.max_dist = max_dist
	def __call__(self, params):
		self.params = params
		for ceph in self.cepheids.T:
		    self._get_ceph_magnitude(ceph)

		return self.cepheids.M, self.cepheids.eM
		cachemaker.clear()

	@lru_cache(maxsize = 1300)
	def _query_tree(self, x):
		indices = self.known.index
		distance, i = self.tree.query(x)
		return distance, indices[i]

	def _get_ceph_magnitude(self, index):
		if index == None:
		    return np.NaN
		cepheid = self.cepheids.loc[index]
		if cepheid.Field == 'lmc':
		    mag = cepheid.m_H - self.params['mu_'+cepheid.Field]+self.params['dz']
		    self.cepheids.loc[index,'M'] =  mag
		    self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
		    return mag
		elif cepheid.Field == 'galaxy':
		    mag = cepheid.m_H - (5 * np.log10(1.0e-3/cepheid.parallax) + 25 - cepheid.LK)+self.params['dz']
		    self.cepheids.loc[index,'M'] =  mag
		    self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
		    return mag

		elif cepheid.Field == 'n4258':
		    mag = cepheid.m_H - self.params['mu_'+cepheid.Field]
		    self.cepheids.loc[index,'M'] =  mag
		    self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
		    return mag
		elif cepheid.Field == 'm31':
		    distance, closest_index = self._query_tree(tuple(cepheid[['T1','T2']]))
		    if distance < self.max_dist:
		    	mag = self._get_ceph_magnitude(closest_index)
		    else:
		    	mag = np.NaN
		    self.cepheids.loc[index,'M'] =  mag
		    self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
		    return mag           
		else:
		    distance, closest_index = self._query_tree(tuple(cepheid[['T1','T2']]))

		    if distance < self.max_dist:
		    	mag = self._get_ceph_magnitude(closest_index)
		    else:
		    	mag = np.NaN
		    self.cepheids.loc[index,'M'] =  mag
		    self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
		    return mag


# cachemaker = CacheMaker()
# class ClosestNeighborMagnitude(object):
#     def __init__(self, cepheids, max_dist = .5):
#         cepheids['logP'] = cepheids.Period.apply(np.log10)
#         cepheids['eM'] = 0
#         self.cepheids = cepheids
#         self.metric = inv(self.cepheids[['logP', 'VminusI']].cov())
#         self.known = self.cepheids.loc[self.cepheids.Field.isin(['lmc','galaxy', 'n4258'])]
#         self.unknown = self.cepheids.loc[~self.cepheids.Field.isin(['lmc','galaxy', 'n4258'])]
#         self.max_dist = max_dist
#         print 'building hash'
#         self.hash = {ceph: self._get_closest_known_index(ceph) for ceph in self.unknown.T}
        
#     def __call__(self, params):
#         self.params = params
#         for ceph in self.cepheids.T:
#             self._get_ceph_magnitude(ceph)

#         return self.cepheids.M, self.cepheids.eM
#         cachemaker.clear()
    
#     def _get_closest_known_index(self, index):
#         k_index = self.known.index[np.argmin([self._get_distance(
#                         self.cepheids.T[index],
#                         self.known.T[k_index]
#                         ) for k_index in self.known.T])]
#         distance = self._get_distance(
#             self.cepheids.T[index],
#             self.known.T[k_index]
#         )
#         if distance < self.max_dist:
#             return k_index
#         else:
#             return None
            
#     @cachemaker.lrucache(maxsize = 1000, name = 'magnitudes')       
#     def _get_ceph_magnitude(self, index):
#         if index == None:
#             return np.NaN
#         cepheid = self.cepheids.loc[index]
#         if cepheid.Field == 'lmc':
#             mag = cepheid.m_H - self.params['mu_'+cepheid.Field]+params['dz']
#             self.cepheids.loc[index,'M'] =  mag
#             self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
#             return mag
#         elif cepheid.Field == 'galaxy':
#             mag = cepheid.m_H - (5 * np.log10(1.0e-3/cepheid.parallax) + 25 - cepheid.LK)+params['dz']
#             self.cepheids.loc[index,'M'] =  mag
#             self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
#             return mag
        
#         elif cepheid.Field == 'n4258':
#             mag = cepheid.m_H - self.params['mu_'+cepheid.Field]
#             self.cepheids.loc[index,'M'] =  mag
#             self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
#             return mag
#         elif cepheid.Field == 'm31':
#             closest_ceph = self.hash[index]
#             mag = self._get_ceph_magnitude(closest_ceph)
#             self.cepheids.loc[index,'M'] =  mag
#             self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
#             return mag           
#         else:
#             closest_ceph = self.hash[index]
#             mag = self._get_ceph_magnitude(closest_ceph)
#             self.cepheids.loc[index,'M'] =  mag
#             self.cepheids.loc[index,'eM'] = np.sqrt(self.cepheids.loc[index,'eM']**2 + cepheid.err**2)
#             return mag
        
#     def _get_distance(self, x, y):
#         x = x[['logP','VminusI']].values
#         y = y[['logP','VminusI']].values
#         return np.sqrt(np.dot(x-y, np.dot(self.metric, (x-y).T)))







def preprocess_cepheids(cepheids):
	return cepheids


def get_data():
    '''
    Grabs the cepheids and sne from the R16 sample
    returns: 
        cepheids: Cepheid dataframe
        sne: sne dataframe
    '''
    filename = '../data/r16_table4.out'
    sne_start = 40
    sne_end = 59
    sne_lines = np.arange(sne_start,sne_end)
    sne = pd.DataFrame(columns = ['Field', 'sne', 'm_0B', 'err'], index = np.arange(sne_end - sne_start))
    ceph_start = 70
    ceph_end = 2346
    cepheid_lines = np.arange(ceph_start,ceph_end)
    cepheids = pd.DataFrame(columns = ['Field','RA','DEC','ID','Period','VminusI','m_H','sigma_tot','Z'], 
                            index = np.arange(ceph_end - ceph_start),
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
    cepheids['err'] = np.sqrt(cepheids.sigma_tot**2 + (cepheids.p_err / cepheids.parallax * 5/np.log(10))**2)
    return cepheids, sne



if __name__ == '__main__':
	analysis = Model()
	[anlaysis() for _ in 1000]

