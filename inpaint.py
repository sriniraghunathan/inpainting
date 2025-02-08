import numpy as np, os, flatsky, tools
import scipy as sc
from pylab import *

#################################################################################
#################################################################################
#################################################################################

def calccov(sim_mat, noofsims, npixels):
    
    m = sim_mat.flatten().reshape(noofsims,npixels)
    m = np.mat( m ).T
    mt = m.T

    cov = (m * mt) / (noofsims)# - 1)
    return cov

#################################################################################
def get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer, square = 0, in_arcmins = 1):

    if not in_arcmins:
        ra_grid = ra_grid * 60.
        dec_grid = dec_grid * 60.

    if not square:
        radius = np.sqrt( (ra_grid**2. + dec_grid**2.) )
        inds_inner = np.where((radius<=mask_radius_inner))
        inds_outer = np.where((radius>mask_radius_inner) & (radius<=mask_radius_outer) )
    else:
        inds_inner = np.where( (abs(ra_grid)<=mask_radius_inner) & (abs(dec_grid)<=mask_radius_inner) )
        ##inds_outer = np.where( (abs(ra_grid_arcmins)>mask_radius_inner) & (abs(ra_grid_arcmins)<=mask_radius_outer) & (abs(dec_grid_arcmins)>mask_radius_inner) & (abs(dec_grid_arcmins)<=mask_radius_outer))
        inds_outer = np.where( (abs(ra_grid)<=mask_radius_outer) & (abs(dec_grid)<=mask_radius_outer) & ( (abs(ra_grid)>mask_radius_inner) | (abs(dec_grid)>mask_radius_inner) ) )

    return inds_inner, inds_outer

#################################################################################
def get_covariance(ra_grid, dec_grid, mapparams, el, cl_dic, bl, nl_dic, noofsims, mask_radius_inner, mask_radius_outer, low_pass_cutoff = 1, maxel_for_grad_filter = None):

    print('\n\tcalculating the covariance from simulations for inpainting')

    ############################################################
    #get the low pass filter
    if low_pass_cutoff:
        assert mask_radius_inner is not None
        if maxel_for_grad_filter is None:
            maxel_for_grad_filter = int( 3.14/np.radians(mask_radius_inner/60.) )
        print(maxel_for_grad_filter)
        lpf = flatsky.get_lpf_hpf(mapparams, maxel_for_grad_filter, filter_type = 0)

    ############################################################
    do_pol = 0
    if 'EE' in cl_dic:
        do_pol = 1
        assert 'TT' in cl_dic and 'EE' in cl_dic and 'TE' in cl_dic
        assert 'T' in nl_dic and 'P' in nl_dic
    #get the sims for covariance calculation
    print('\n\t\tgenerating %s sims' %(noofsims))
    sims_for_covariance = []
    for n in range(noofsims):

        #cmb sim and beam, for CMB include the transfer function and beam
        if do_pol:
            cmb_map = flatsky.make_gaussian_realisation(mapparams, el, cl_dic['TT'], cl2 = cl_dic['EE'], cl12 = cl_dic['TE'], bl = bl)
        else:
            cmb_map = flatsky.make_gaussian_realisation(mapparams, el, cl_dic['TT'], bl = bl)

        #noise map, do not include the beam or transfer function
        if do_pol:
            noise_map_T = flatsky.make_gaussian_realisation(mapparams, el, nl_dic['T'])
            noise_map_Q = flatsky.make_gaussian_realisation(mapparams, el, nl_dic['P'])
            noise_map_U = flatsky.make_gaussian_realisation(mapparams, el, nl_dic['P'])
            noise_map = np.asarray( [noise_map_T, noise_map_Q, noise_map_U] )
        else:
            noise_map = flatsky.make_gaussian_realisation(mapparams, el, nl_dic['T'])
        sim_map = cmb_map + noise_map
        #imshow(sim_map);colorbar(); show(); sys.exit()

        ############################################################
        #lpf the map
        if low_pass_cutoff:
            sim_map = np.fft.ifft2( np.fft.fft2(sim_map) * lpf ).real
        #imshow(sim_map);colorbar(); show(); sys.exit()

        sims_for_covariance.append( sim_map )
    sims_for_covariance = np.asarray( sims_for_covariance)

    ############################################################
    #get the inner and outer pixel indices
    inds_inner, inds_outer = get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)

    ############################################################
    #get the pixel values in the inner and outer regions
    if do_pol:
        t1_for_cov = sims_for_covariance[:, :, inds_inner[0], inds_inner[1]]
        t2_for_cov = sims_for_covariance[:, :, inds_outer[0], inds_outer[1]]
        npixels_t1_tmp = t1_for_cov.shape[-1]
        npixels_t2_tmp = t2_for_cov.shape[-1]
        #contains T/Q/U vector concatenated
        t1_for_cov = t1_for_cov.reshape( noofsims, 3 * npixels_t1_tmp)
        t2_for_cov = t2_for_cov.reshape( noofsims, 3 * npixels_t2_tmp)
    else:
        t1_for_cov = sims_for_covariance[:,inds_inner[0], inds_inner[1]]
        t2_for_cov = sims_for_covariance[:,inds_outer[0], inds_outer[1]]

    ############################################################
    #get the covariance now
    npixels_t1 = t1_for_cov.shape[1]
    npixels_t2 = t2_for_cov.shape[1]

    t1t2_for_cov = np.concatenate( (t1_for_cov,t2_for_cov), axis = 1 )
    npixels_t1t2 = t1t2_for_cov.shape[1]
    t1t2_cov = calccov(t1t2_for_cov, noofsims, npixels_t1t2)

    #print(t1t2_for_cov.shape, npixels_t1, npixels_t2, npixels_t1t2)
    #print(t1t2_cov.shape)
    #sys.exit()

    #logline = '\t\tcalcuating sigmas now'
    #logfile = open(log_file,'a');logfile.writelines('%s\n' %(logline));logfile.close()
    #print(logline)

    ############################################################
    #https://arxiv.org/pdf/1301.4145.pdf
    ##Eq. 32
    #sigma_11 = t1t2_cov[:npixels_t1, : npixels_t1] 
    sigma_22 = t1t2_cov[npixels_t1:,npixels_t1:]
    sigma_12 = t1t2_cov[:npixels_t1,npixels_t1:]
    #sigma_21 = t1t2_cov[npixels_t1:,:npixels_t1]

    print('\n\t\t\tinvert sigma_22 matrix (%s,%s) now' %(sigma_22.shape[0], sigma_22.shape[1]))
    sigma_22_inv = sc.linalg.pinv2(sigma_22)
    sigma_dic = {}
    ##sigma_dic['sigma_11'] = sigma_11
    #sigma_dic['sigma_11_inv'] = sigma_11_inv
    ##sigma_dic['sigma_22'] = sigma_22
    sigma_dic['sigma_22_inv'] = sigma_22_inv
    sigma_dic['sigma_12'] = sigma_12
    ##sigma_dic['sigma_21'] = sigma_21

    print('\n\t\tcovariance obtained')
    return sigma_dic

#################################################################################

def inpainting(map_dic_to_inpaint, ra_grid, dec_grid, mapparams, el, cl_dic, bl, nl_dic, noofsims, mask_radius_inner, mask_radius_outer, low_pass_cutoff = 1, intrp_r1_before_lpf = 0, mask_inner = 0, sigma_dic = None, maxel_for_grad_filter=None, use_original=False, use_cons_gau_sims = True):

    #print('\n\tperform inpainting')
    """
    mask_inner = 1: The inner region is masked before the LPF. Might be useful in the presence of bright SZ signal at the centre.
    """

    ############################################################
    #get covariance
    if sigma_dic is None:
        #get covariance for inpainting
        sigma_dic = get_covariance(ra_grid, dec_grid, mapparams, el, cl_dic, bl, nl_dic, noofsims, mask_radius_inner, mask_radius_outer, low_pass_cutoff=low_pass_cutoff, cltwod=cltwod, tf=tf, nltwod=nltwod, maxel_for_grad_filter=maxel_for_grad_filter)
    sigma_12 = sigma_dic['sigma_12']
    sigma_22_inv = sigma_dic['sigma_22_inv']

    ############################################################
    #get the low pass filter
    if low_pass_cutoff:
        assert mask_radius_inner is not None
        if maxel_for_grad_filter is None:
            maxel_for_grad_filter = int( 3.14/np.radians(mask_radius_inner/60.) )
        lpf = flatsky.get_lpf_hpf(mapparams, maxel_for_grad_filter, filter_type = 0)

    ############################################################
    #get the inner and outer pixel indices
    inds_inner, inds_outer = get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)

    ############################################################
    #determine if just T or T/Q/U
    do_pol = 0
    tqukeys = ['T']
    if 'Q' in map_dic_to_inpaint:
        do_pol = 1
        assert 'TT' in cl_dic and 'EE' in cl_dic and 'TE' in cl_dic
        assert 'T' in nl_dic and 'P' in nl_dic
        assert 'T' in map_dic_to_inpaint and 'Q' in map_dic_to_inpaint and 'U' in map_dic_to_inpaint
        tqukeys = ['T', 'Q', 'U']

    map_to_inpaint = []
    for tqu in tqukeys:
        map_to_inpaint.append( map_dic_to_inpaint[tqu] )
    map_to_inpaint = np.asarray( map_to_inpaint )

    original_map = map_to_inpaint.copy()
    ############################################################
    #mask the inner region before LPF if required - might be useful in the presence of bright SZ signal at the centre
    if mask_inner and low_pass_cutoff: #otherwise t2 will see the HPF artefact
        print('\n\n\t\tnot yet implemented\n\n')
        sys.exit()
        mask = masking_for_filtering(ra_grid, dec_grid, simmaps, mask_radius = mask_radius_inner - 2.)
        map_to_inpaint = map_to_inpaint * mask

    ############################################################
    if intrp_r1_before_lpf:
        for i in range(len(tqukeys)):
            map_to_inpaint[i] = inpterpolate_r1_using_r2(map_to_inpaint[i], ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)
    ############################################################

    #lpf the map
    if low_pass_cutoff:
        map_to_inpaint = np.fft.ifft2( np.fft.fft2(map_to_inpaint) * lpf).real
        ##imshow(map_to_inpaint);colorbar();show();sys.exit()
    ############################################################
    #get the pixel values in the inner and outer regions 
    t1_data = map_to_inpaint[:, inds_inner[0], inds_inner[1]].flatten()
    t2_data = map_to_inpaint[:, inds_outer[0], inds_outer[1]].flatten()
    #print(t1_data.shape, t2_data.shape)
    #sys.exit()

    ############################################################
    #generate constrained Gaussia CMB realisation now
    # include transfer function for the data map
    # include noise twod for the noise

    '''
    cmb_map = flatsky.make_gaussian_realisation(mapparams, el, cl, bl = bl) #cmb sim and beam
    noise_map = flatsky.make_gaussian_realisation(mapparams, el, nl) #noise map
    '''
    if use_cons_gau_sims:
        if do_pol:
            cmb_map = flatsky.make_gaussian_realisation(mapparams, el, cl_dic['TT'], cl2 = cl_dic['EE'], cl12 = cl_dic['TE'], bl = bl)
            noise_map_T = flatsky.make_gaussian_realisation(mapparams, el, nl_dic['T'])
            noise_map_Q = flatsky.make_gaussian_realisation(mapparams, el, nl_dic['P'])
            noise_map_U = flatsky.make_gaussian_realisation(mapparams, el, nl_dic['P'])
            noise_map = np.asarray( [noise_map_T, noise_map_Q, noise_map_U] )
        else:
            cmb_map = np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, cl_dic['TT'], bl = bl)] )
            noise_map = np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, nl_dic['T'])] )

        constrained_sim_to_inpaint = cmb_map + noise_map #combined
        #lpf the map
        if low_pass_cutoff:
            constrained_sim_to_inpaint = np.fft.ifft2( np.fft.fft2(constrained_sim_to_inpaint) * lpf ).real

        if (0):
            tr, tc = 1, len(tqukeys)
            for tqucntr in range( tc ):
                subplot(tr, tc, tqucntr + 1); imshow(constrained_sim_to_inpaint[tqucntr]); colorbar(); 
            show(); 
            #sys.exit()
    else: #20241031
        constrained_sim_to_inpaint = np.zeros_like(map_to_inpaint)

    ############################################################
    #get the pixel values in the inner and outer regions from the constrained realisation
    #t1_tilde = constrained_sim_to_inpaint[inds_inner[0], inds_inner[1]]
    #t2_tilde = constrained_sim_to_inpaint[inds_outer[0], inds_outer[1]]
    t1_tilde = constrained_sim_to_inpaint[:, inds_inner[0], inds_inner[1]].flatten()
    t2_tilde = constrained_sim_to_inpaint[:, inds_outer[0], inds_outer[1]].flatten()
    #print(t1_tilde.shape, t2_tilde.shape)

    ############################################################
    #get the modified t1 values
    inpainted_t1 = np.asarray( t1_tilde + np.dot(sigma_12, np.dot(sigma_22_inv, ( t2_data - t2_tilde) ) ) )[0]  ##Eq. 36
    #print(inpainted_t1.shape)
    ############################################################
    #create a new inpainted map: copy the old map and replace the t1 region
    # if use_original is True, put the inpainted region in the input map before lpf
    if use_original:
        map_to_inpaint = original_map
    inpainted_map = np.copy(map_to_inpaint)

    #inpainted_map[inds_inner[0], inds_inner[1]] = inpainted_t1
    #split inpainted T(or /Q/U) vector in nx1(or 3) array
    npixels_t1 = int( len(t1_tilde) / len(tqukeys) )
    inpainted_t1_tqu_split = inpainted_t1.reshape( len(tqukeys), npixels_t1 )
    inpainted_map[:, inds_inner[0], inds_inner[1]] = inpainted_t1_tqu_split
    if (0):
        print(np.allclose(inpainted_map[:, inds_inner[0], inds_inner[1]], inpainted_t1_tqu_split))
        #print(np.allclose(inpainted_map[0, inds_inner[0], inds_inner[1]], inpainted_t1_tqu_split[0]))
        #print(np.allclose(inpainted_map[1, inds_inner[0], inds_inner[1]], inpainted_t1_tqu_split[1]))
        #print(np.allclose(inpainted_map[2, inds_inner[0], inds_inner[1]], inpainted_t1_tqu_split[2]))
        sys.exit()

    if (0):
        #subplot(121);imshow(map_to_inpaint);colorbar(); subplot(122);imshow(inpainted_map);colorbar();show();sys.exit()
        tr, tc = len(tqukeys), 2
        sbpl = 1
        for tqucntr in range( len(tqukeys) ):
            subplot(tr, tc, sbpl); imshow(map_to_inpaint[tqucntr]); colorbar(); title(r'%s' %(tqukeys[tqucntr]))
            sbpl += 1
            subplot(tr, tc, sbpl); imshow(inpainted_map[tqucntr]); colorbar(); title(r'$\tilde{%s}$' %(tqukeys[tqucntr]))
            sbpl += 1
        show()
        #sys.exit() 

    cmb_inpainted_map = np.copy(map_to_inpaint)*0.
    #cmb_inpainted_map[inds_inner[0], inds_inner[1]] = inpainted_t1
    cmb_inpainted_map[:, inds_inner[0], inds_inner[1]] = inpainted_t1_tqu_split

    return cmb_inpainted_map, inpainted_map, map_to_inpaint

#################################################################################

def masking_for_filtering(ra_grid, dec_grid, mask_radius = 2., taper_radius = 6., in_arcmins = 1):

    import scipy as sc
    import scipy.ndimage as ndimage

    if not in_arcmins:
        ra_grid_arcmins = ra_grid * 60.
        dec_grid_arcmins = dec_grid * 60.

    radius = np.sqrt( (ra_grid_arcmins**2. + dec_grid_arcmins**2.) )

    mask = np.ones( ra_grid_arcmins.shape )
    if (1): #20180118
        ##print '\n\n\t\t fixing masking radius to %s\n\n' %mask_ra_gridDIUS_ARCMINS
        inds_to_mask = np.where((radius<=mask_radius)) #2arcmins - fix this for now
        mask[inds_to_mask[0], inds_to_mask[1]] = 0.

    ker=np.hanning(taper_radius)
    ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )

    mask=ndimage.convolve(mask, ker2d)
    mask/=mask.max()

    return mask
