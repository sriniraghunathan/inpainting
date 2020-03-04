import numpy as np, os, flatsky
import scipy as sc
import scipy.ndimage as ndimage

from pylab import *

#################################################################################
#################################################################################
#################################################################################

def get_bl(beamval, el, make_2d = 0, mapparams = None):

    fwhm_radians = np.radians(beamval/60.)
    sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
    sigma2 = sigma ** 2
    #bl = np.exp(el * (el+1) * sigma2)
    bl = np.exp(-0.5 * el * (el+1) * sigma2)

    if make_2d:
        assert mapparams is not None
        el = np.arange(len(bl))
        bl = flatsky.cl_to_cl2d(el, bl, mapparams) 

    return bl

################################################################################################################

def get_nl(noiseval, el, beamval = None, use_beam_window = 0, uk_to_K = 0, elknee_t = -1, alpha_knee = 0):

    if uk_to_K: noiseval = noiseval/1e6

    if use_beam_window:
        fwhm_radians = np.radians(beamval/60.)
        sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
        sigma2 = sigma ** 2
        bl = np.exp(el * (el+1) * sigma2)

    delta_T_radians = noiseval * np.radians(1./60.)
    nl = np.tile(delta_T_radians**2., int(max(el)) + 1 )

    nl = np.asarray( [nl[int(l)] for l in el] )

    if use_beam_window: nl *= bl

    if elknee_t != -1.:
        nl = np.copy(nl) * (1. + (elknee_t * 1./el)**alpha_knee )

    return nl

################################################################################################################