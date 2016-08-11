from astropy.utils.data import download_file
import pandas as pd



class Data(object):
	def __init__(self):
		cepheid_table = download_file(
			'http://iopscience.iop.org/0004-637X/730/2/119/suppdata/apj383673t2_mrt.txt', 
			cache = True
									 )

		self.cepheids = pd.read_csv(
							   cepheid_table,
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


		self.cepheids=self.cepheids.fillna(value = '-')

		sne_table = download_file(
		        'http://iopscience.iop.org/0004-637X/730/2/119/suppdata/apj383673t3_ascii.txt',
		        cache = True)
		self.sne = pd.read_csv(sne_table, 
		                   
		                  delim_whitespace=True, 
		                  skiprows = [0,1,2,3,4,13,15],
		                  names = (['Host', 'junk','Sn1A',
		                            'Filters', 'm0_viPlus5a_v',
		                            'sigma', 'DeltaMu_0','e_DeltaMu_0',
		                            'mu_0_Best', 'e_mu_0_Best'
		                          ])
		                 )
		self.sne.loc[:,'e_DeltaMu_0'] = (self.sne.loc[:,'e_DeltaMu_0'].apply(str).str.replace('\(|\)','')).astype('float')
		self.sne.loc[:,'e_mu_0_Best'] = (self.sne.loc[:,'e_mu_0_Best'].apply(str).str.replace('\(|\)','')).astype('float')


	def __call__(self):
		print self.sne
		print self.cepheids

	