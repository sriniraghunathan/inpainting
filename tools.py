import numpy as np, os, flatsky
import scipy as sc
import scipy.ndimage as ndimage

from pylab import *

#################################################################################
#################################################################################
#################################################################################

def get_blsqinv(beamval, el, make_2d = 0, mapparams = None):

    """
    Get the inverse of the beam window function sqaured.
   
    Parameters
    ----------
    beamval: float
        Beam FWHM in arcmins.
    el: array
        Multipoles over which the window function must be defined.
    make_2d: bool
        Convert to 2D if desired.
        Default is False.
    mapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    Returns
    -------
    blsqinv: array.
        1/Bl^2 either in 1d or 2D.
    """

    fwhm_radians = np.radians(beamval/60.)
    sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
    sigma2 = sigma ** 2
    blsqinv = np.exp(-0.5 * el * (el+1) * sigma2)

    if make_2d:
        assert mapparams is not None
        el = np.arange(len(blsqinv))
        blsqinv = flatsky.cl_to_cl2d(el, blsqinv, mapparams) 

    return blsqinv

################################################################################################################

def get_nl(noiseval_in_ukarcmin, el, beamval = None, elknee_t = -1, alpha_knee = 0):

    """
    Get the noise power spectra: White and 1/f spectrum.
    Can return beam deconvolved nl if desired.

    .. math::
        P(f) = A^2 \\left[ 1+ \\left( \\frac{\\ell_{\\rm knee}}{\\ell}\\right)^{\\alpha_{\\rm knee}} \\right].

    Parameters
    ----------
    noiseval_in_ukarcmin: float
        White noise level in uK-arcmin.
    el: array
        Multipoles over which the window function must be defined.
    beamval: float
        Beam FWHM in arcmins.
        Default is None.
        If supplied, bl^2 will be divided from nl.
    elknee_t: float
        Knee frequency for 1/f  (el_knee in the above equation).
    alpha_knee: float
        Slope for 1/f (alpha_knee in the above equation).

    Returns
    -------
    nl: array.
        (Beam deconvoled) noise power spectrum.
    """

    if beamval is not None:
        fwhm_radians = np.radians(beamval/60.)
        sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
        sigma2 = sigma ** 2
        bl = np.exp(el * (el+1) * sigma2)

    delta_T_radians = noiseval_in_ukarcmin * np.radians(1./60.)
    nl = np.tile(delta_T_radians**2., int(max(el)) + 1 )

    nl = np.asarray( [nl[int(l)] for l in el] )

    if use_beam_window: nl *= bl

    if elknee_t != -1.:
        nl = np.copy(nl) * (1. + (elknee_t * 1./el)**alpha_knee )

    return nl

################################################################################################################