import numpy as np, os, flatsky
import scipy as sc
import scipy.ndimage as ndimage


#################################################################################
#################################################################################
#################################################################################

def fn_get_gaussian_realisation_from_cl(mapparams, el, cl, cl2 = None, cl12 = None):

    nx, ny, dx, dy = mapparams
    arcmins2radians = np.radians(1/60.)

    dx *= arcmins2radians
    dy *= arcmins2radians

    ################################################
    #map stuff
    norm = np.sqrt(1./ (dx * dy))
    ################################################

    #1d to 2d now
    cltwod = flatsky.cl_to_cltwod(np.asarray([el, cl]).T, mapparams)[0]

    ################################################
    if cl2 is not None: #for TE, etc. where two fields are correlated.
        assert cl12 is not None
        cltwod12 = flatsky.cltwod2cltwod(np.asarray([el, cl12]).T, mapparams)[0]
        cltwod2 = flatsky.cltwod2cltwod(np.asarray([el, cl2]).T, mapparams)[0]

    ################################################
    if cl2 is None:

        cltwod = cltwod**0.5 * scalefac
        cltwod[np.isnan(cltwod)] = 0.

        gauss_reals = np.random.standard_normal([nx,ny])
        SIM = np.fft.ifft2( np.copy( cltwod ) * np.fft.fft2( gauss_reals ) ).real

    else: #for TE, etc. where two fields are correlated.

        cltwod12[np.isnan(cltwod12)] = 0.
        cltwod2[np.isnan(cltwod2)] = 0.

        gauss_reals_1 = np.random.standard_normal([nx,ny])
        gauss_reals_2 = np.random.standard_normal([nx,ny])

        gauss_reals_1 = np.fft.fft2( gauss_reals_1 )
        gauss_reals_2 = np.fft.fft2( gauss_reals_2 )

        t1 = gauss_reals_1 * cltwod12 / cltwod2**0.5
        t2 = gauss_reals_2 * ( cltwod - (cltwod12**2. /cltwod2) )**0.5

        SIM_FFT = (t1 + t2) * scalefac
        SIM_FFT[np.isnan(SIM_FFT)] = 0.
        SIM = np.fft.ifft2( SIM_FFT ).real

    SIM = SIM - np.mean(SIM)

    return SIM

################################################################################################################

def get_bl(beamval, el):

    fwhm_radians = np.radians(beamval/60.)
    sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
    sigma2 = sigma ** 2
    bl = np.exp(el * (el+1) * sigma2)

    return bl

################################################################################################################

def get_nl(noiseval, el, beamval, use_beam_window = 1, uk_to_K = 0, elknee_t = -1, alpha_knee = 0):

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
