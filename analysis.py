from numpy import polyfit
from astropy.utils.data import download_file
from astropy.utils.data import Conf 
Conf.remote_timeout = 10
import warnings
warnings.filterwarnings('ignore')
from classes import Cephied, Cephied_Collection, PL_function

class Analysis(object):
    def __init__(self):
        filename = download_file('http://iopscience.iop.org/0004-637X/730/2/119/suppceph/apj383673t2_mrt.txt', cache = True)
        ceph = genfromtxt(filename, 
                       skip_header = 39, 
                        invalid_raise=False,
                        missing_values='',
                        usemask=False,
                        filling_values=0.0,
                        dtype = None);
        names = ['Field', 'RAdeg', 'DEdeg', 'ID',  'Period',  'Color',  'mag', 'e_mag', 'Offset', 'Bias', 'IMrms', 'Z']
        ceph = [dict(zip(names, x)) for x in ceph]
        calib_ceph = Cepheid_Collection([Cepheid(**x) for x in ceph if x['Field'] == 'n4258'])
        ceph = Cepheid_Collection([Cepheid(**x) for x in ceph])
        self.cephieds = ceph


        filename = download_file('http://iopscience.iop.org/0004-637X/730/2/119/suppceph/apj383673t3_ascii.txt', cache = True)
        sne = genfromtxt(filename,
                          skip_header = 4,
                          invalid_raise = False,
                          missing_values = '',
                          usemask=False,
                          filling_values = 0.0,
                          dtype = None
                         );
        names = ['Field', 'junk','Sn1a','Filters','mag_av', 'sigma', 'delta_mag','dmag_error','mu_0_Best','mu_best_error']
        sne = [dict(zip(names, x)) for x in sne]
        sne = Sne_Collection([Sne(**x) for x in sne])
        self.SNe = sne

    def __call__(self, **kwargs):
        self.pl_function = PL_Relation(ceph, **kwargs)


def __main__():
    pl_func = PL_function()
    return pl_func()
