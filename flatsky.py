import numpy as np, sys, os, scipy as sc, healpy as H

################################################################################################################
#flat-sky routines
################################################################################################################

def cl_to_cl2d(el, cl, flatskymapparams, left = 0., right = 0.):

    """
    Interpolating a 1d power spectrum (cl) defined on multipoles (el) to 2D assuming azimuthal symmetry (i.e:) isotropy.

    Parameters
    ----------
    el: array
        Multipoles over which the power spectrium is defined.
    cl: array
        1d power spectrum that needs to be interpolated on the 2D grid.
    flatskymyapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.
    left: float
        value to be used for interpolation outside of the range (lower side).
        default is zero.
    right: float
        value to be used for interpolation outside of the range (higher side).
        default is zero.

    Returns
    -------
    cl2d: array, shape is (ny, nx).
        interpolated power spectrum on the 2D grid.
    """


    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape) 

    return cl2d

################################################################################################################

def get_lxly(flatskymapparams):

    """
    return lx, ly modes (kx, ky Fourier modes) for a flatsky map grid.
    
    Parameters
    ----------
    flatskymyapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    Returns
    -------
    lx, ly: array, shape is (ny, nx).
    """

    nx, ny, dx, dx = flatskymapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

################################################################################################################

def get_lxly_az_angle(lx,ly):

    """
    azimuthal angle from lx, ly

    Parameters
    ----------
    lx: array
        lx modes

    ly: array
        ly modes

    Returns
    -------
    phi: array
        azimuthal angle
    """

    phi = 2*np.arctan2(lx, -ly)
    return phi

################################################################################################################
def convert_eb_qu(map1, map2, flatskymapparams, eb_to_qu = 1):

    """
    Convert EB/QU into each other.

    Parameters
    ----------
    map1: array
        flatsky map of E or Q.

    map2: array
        flatsky map of B or U.

    flatskymyapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    eb_to_qu: bool
        Either EB-->QU or QU-->EB.
        Default is EB-->QU.

    Returns
    -------
    map1_mod: array
        flatsky map of E or Q.

    map2_mod: array
        flatsky map of B or U.
    """

    lx, ly = get_lxly(flatskymapparams)
    angle = get_lxly_az_angle(lx,ly)

    map1_fft, map2_fft = np.fft.fft2(map1),np.fft.fft2(map2)
    if eb_to_qu:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft - np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real
    else:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft + np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( -np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real

    return map1_mod, map2_mod
################################################################################################################

def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type = 0):

    """
    Get 2D Fourier filters. Supports low-pass (LPF), high-pass (HPF), and band-pass (BPF) filters.

    Parameters
    ----------
    flatskymyapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    lmin_lmax: list
        Contains lmin and lmax values for the filters.
        For low-pass (LPF), lmax = lmin_lmax[0].
        For high-pass (HPF), lmin = lmin_lmax[0].
        For band-pass (BPF), lmin, lmax = lmin_lmax.

    filter_type: int
        0: LPF
        1: HPF
        2: BPF
        Default is LPF.

    Returns
    -------
    fft_filter: array
        Requested 2D Fourier filter.
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        lmax = lmin_lmax[0]
        fft_filter[ell>lmax] = 0.
    elif filter_type == 1:
        lmin = lmin_lmax[0]
        fft_filter[ell<lmin] = 0.
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell<lmin] = 0.
        fft_filter[ell>lmax] = 0

    return fft_filter

################################################################################################################

def wiener_filter(flatskymyapparams, cl_signal, cl_noise, el = None):

    """
    Get 2D Wiener filter.

    .. math::
        W(\\ell) = \\frac{ C_{\\ell}^{\\rm signal} } {C_{\\ell}^{\\rm signal} + C_{\\ell}^{\\rm noise}}


    Parameters
    ----------
    flatskymyapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    cl_signal: array
        Power spectrum of the signal component.

    cl_noise: array
        Power spectrum of the noise component.

    el: array (optional)
        Multipole over which the signal / noise spectra are defined.
        Default is None and el will be np.arange( len(cl_signal) )

    Returns
    -------
    wiener_filter: array
        2D Wiener filter.
    """

    if el is None: el = np.arange(len(cl_signal))

    nx, ny, dx, dx = flatskymapparams

    #get 2D cl
    cl_signal2d = cl_to_cl2d(el, cl_signal, flatskymapparams) 
    cl_noise2d = cl_to_cl2d(el, cl_noise, flatskymapparams) 

    wiener_filter = cl_signal2d / (cl_signal2d + cl_noise2d)

    return wiener_filter

################################################################################################################

def cl2map(flatskymapparams, cl, el = None):

    """
    cl2map module - creates a flat sky map based on the flatskymap parameters and the input power spectra.
    Look into make_gaussian_realisation for a more general code. 

    Parameters
    ----------
    flatskymyapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    cl: array
        1d vector of Cl power spectra: temp / pol. power spectra

    el: array (optional)
        Multipole over which the signal / noise spectra are defined.
        Default is None and el will be np.arange( len(cl_signal) )

    Returns
    -------
    flatskymap: array
        flatskymap with the given underlying power spectrum cl.

    See Also
    -------
    make_gaussian_realisation
    """

    if el is None: el = np.arange(len(cl))

    nx, ny, dx, dx = flatskymapparams

    #get 2D cl
    cl2d = cl_to_cl2d(el, cl, flatskymapparams) 

    #pixel area normalisation
    dx_rad = np.radians(dx/60.)
    pix_area_norm = np.sqrt(1./ (dx_rad**2.))
    cl2d_sqrt_normed = np.sqrt(cl2d) * pix_area_norm

    #make a random Gaussian realisation now
    gauss_reals = np.random.randn(nx,ny)
    
    #convolve with the power spectra
    flatskymap = np.fft.ifft2( np.fft.fft2(gauss_reals) * cl2d_sqrt_normed).real
    flatskymap = flatskymap - np.mean(flatskymap)

    return flatskymap    

################################################################################################################

def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None):

    """
    map2cl module - get the auto-/cross-power spectra of map/maps

    Parameters
    ----------
    flatskymyapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    flatskymap1: array
        flatskymap1 with dimensions (ny, nx).

    flatskymap2: array (Optional)
        flatskymap2 with dimensions (ny, nx).
        Default is None.
        If None, compute the auto-spectrum of flatskymap1.
        If not None, compute the cross-spectrum between flatskymap1 and flatskymap2.

    binsize: int
        el bins. computed automatically based on the fft grid spacing if None.

    Returns
    -------
    el: array
        Multipoles over which the power spectrum is defined.
    cl: array
        auto/cross power spectrum.
    """

    nx, ny, dx, dx = flatskymapparams
    dx_rad = np.radians(dx/60.)

    lx, ly = get_lxly(flatskymapparams)

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)

    rad_prf = radial_profile(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
    el, cl = rad_prf[:,0], rad_prf[:,1]

    return el, cl

################################################################################################################

def radial_profile(z, xy = None, bin_size = 1., minbin = 0., maxbin = 10., to_arcmins = 1, get_errors = 1):

    """
    get the radial profile of an image (both real and fourier space).
    Can be used to compute radial profile of stacked profiles or 2D power spectrum.

    Parameters
    ----------
    z: array
        image to get the radial profile.
    xy: array
        x and y grid. Same shape as the image z.
        Default is None.
        If None, 
        x, y = np.indices(image.shape)
    bin_size: float
        radial binning factor.
        default is 1.
    minbin: float
        minimum bin for radial profile
        default is 0.
    maxbin: float
        minimum bin for radial profile
        default is 10.
    to_arcmins: bool
        If set, then xy are assumed to be in degrees and multipled by 60 to convert to arcmins.
    get_errors: bool
        obtain scatter in each bin.
        This is not the error due to variance. Just the sample variance.
        Default is True.

    Returns
    -------
    radprf: array.
        Array with three elements cotaining
        radprf[:,0] = radial bins
        radprf[:,1] = radial binned values
        if get_errors:
        radprf[:,2] = radial bin errors.
    """

    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(image.shape)
    else:
        x, y = xy

    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
    if to_arcmins: radius *= 60.

    binarr=np.arange(minbin,maxbin,bin_size)
    radprf=np.zeros((len(binarr),3))

    hit_count=[]

    for b,bin in enumerate(binarr):
        ind=np.where((radius>=bin) & (radius<bin+bin_size))
        radprf[b,0]=(bin+bin_size/2.)
        hits = len(np.where(abs(z[ind])>0.)[0])

        if hits>0:
            radprf[b,1]=np.sum(z[ind])/hits
            radprf[b,2]=np.std(z[ind])
        hit_count.append(hits)

    hit_count=np.asarray(hit_count)
    std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
    errval=std_mean/(hit_count)**0.5
    if get_errors:
        radprf[:,2]=errval

    return radprf

################################################################################################################

def make_gaussian_realisation(mapparams, el, cl, cl2 = None, cl12 = None, cltwod=None, tf=None, bl = None, qu_or_eb = 'qu'):

    """
    Make gaussian realisation of flat sky map or 2maps based on the flatskymap parameters and the input power spectra.
    Look into cl2map for a simple version.

    Parameters
    ----------
    flatskymyapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.
    el: array
        Multipoles over which the power spectrum is defined.
    cl: array
        1d vector of Cl auto-power spectra for map1.
    cl2: array (optional)
        1d vector of Cl2 auto-power spectra for map2.
        Default is None. Used to generate correlated maps.
    cl12: array (optional)
        1d vector of Cl2 cross-power spectra of map1 and map2.
        Default is None. Used to generate correlated maps.
    cltwod: array
        2D version of cl. 
        Default is None. Computed using 1d vector assuming azimuthal symmetry.
    tf: array
        2D filtering. 
        Default is None. Used to removed filtered modes.
    bl: array
        1d beam window function. 
        Default is None. Used for smoothing the maps.
    qu_or_eb: array
        Generates TQU or TEB maps if cl, cl2, cl12 are supplied.
        Default is 'QU'.

    Returns
    -------
    sim_map_arr: array.
        sim_map1: T-map.
        if cl2 and cl12 are provided:
        sim_map2: Q or E map.
        sim_map3: U or B map.
        
    See Also
    -------
    cl2map
    """
    nx, ny, dx, dy = mapparams
    arcmins2radians = np.radians(1/60.)

    dx *= arcmins2radians
    dy *= arcmins2radians

    ################################################
    #map stuff
    norm = np.sqrt(1./ (dx * dy))
    ################################################

    #if cltwod is given, directly use it, otherwise do 1d to 2d 
    if cltwod is None:
        cltwod = cl_to_cl2d(el, cl, mapparams)

    # if the tranfer function is given, correct the 2D cl by tf
    if tf is not None:
        cltwod = cltwod * tf**2
    
    ################################################
    if cl2 is not None: #for TE, etc. where two fields are correlated.
        assert cl12 is not None
        cltwod12 = cl_to_cl2d(el, cl12, mapparams)
        cltwod2 = cl_to_cl2d(el, cl2, mapparams)

    ################################################
    if cl2 is None:

        cltwod = cltwod**0.5 * norm
        cltwod[np.isnan(cltwod)] = 0.

        gauss_reals = np.random.standard_normal([nx,ny])
        SIM = np.fft.ifft2( np.copy( cltwod ) * np.fft.fft2( gauss_reals ) ).real

    else: #for TE, etc. where two fields are correlated.

        assert qu_or_eb in ['qu', 'eb']

        cltwod[np.isnan(cltwod)] = 0.
        cltwod12[np.isnan(cltwod12)] = 0.
        cltwod2[np.isnan(cltwod2)] = 0.

        #in this case, generate two Gaussian random fields
        #sim_field_1 will simply be generated from gauss_reals_1 like above
        #sim_field_2 will generated from both gauss_reals_1, gauss_reals_2 using the cross spectra
        gauss_reals_1 = np.random.standard_normal([nx,ny])
        gauss_reals_2 = np.random.standard_normal([nx,ny])

        gauss_reals_1_fft = np.fft.fft2( gauss_reals_1 )
        gauss_reals_2_fft = np.fft.fft2( gauss_reals_2 )

        #field_1
        cltwod_tmp = np.copy( cltwod )**0.5 * norm
        sim_field_1 = np.fft.ifft2( cltwod_tmp *  gauss_reals_1_fft ).real
        #sim_field_1 = np.zeros( (ny, nx) )

        #field 2 - has correlation with field_1
        t1 = np.copy( gauss_reals_1_fft ) * cltwod12 / np.copy(cltwod)**0.5
        t2 = np.copy( gauss_reals_2_fft ) * ( cltwod2 - (cltwod12**2. /np.copy(cltwod)) )**0.5
        sim_field_2_fft = (t1 + t2) * norm
        sim_field_2_fft[np.isnan(sim_field_2_fft)] = 0.
        sim_field_2 = np.fft.ifft2( sim_field_2_fft ).real

        #T and E generated. B will simply be zeroes.
        sim_field_3 = np.zeros( sim_field_2.shape )
        if qu_or_eb == 'qu': #T, Q, U: convert E/B to Q/U.        
            sim_field_2, sim_field_3 = convert_eb_qu(sim_field_2, sim_field_3, mapparams, eb_to_qu = 1)
        else: #T, E, B: B will simply be zeroes
            pass

        sim_map_arr = np.asarray( [sim_field_1, sim_field_2, sim_field_3] )

    if bl is not None:
        if np.ndim(bl) != 2:
            bl = cl_to_cl2d(el, bl, mapparams)
        sim_map_arr = np.fft.ifft2( np.fft.fft2(sim_map_arr) * bl).real

    if cl2 is None:
        sim_map_arr = sim_map_arr - np.mean(sim_map_arr)
    else:
        for tqu in range(len(sim_map_arr)):
            sim_map_arr[tqu] = sim_map_arr[tqu] - np.mean(sim_map_arr[tqu])

    return sim_map_arr

################################################################################################################
