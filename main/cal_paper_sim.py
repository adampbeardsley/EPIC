import datetime as DT
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import scipy.constants as FCNST
import antenna_array as AA
import geometry as GEOM
import sim_observe as SIM
import my_DSP_modules as DSP
import ipdb as PDB
import EPICal
import aperture as APR
import time
import pickle

t1 = time.time()

# @profile
# def main():
cal_iter = 400
itr = 25 * cal_iter
rxr_noise = 0.0
model_frac = 1.0  # fraction of total sky flux to model
make_ideal_cal = True

grid_map_method = 'sparse'
# grid_map_method='regular'

# *** Antenna and array initialization ***

lat = -26.701  # Latitude of MWA in degrees
f0 = 150e6  # Center frequency

nchan = 4
nts = nchan / 2
f_center = f0
channel_width = 40e3
bandwidth = nchan * channel_width
dt = 1 / bandwidth
freqs = NP.arange(f0 - nchan / 2 * channel_width, f0 + nchan / 2 * channel_width, channel_width)

# ** Use this for MWA core
antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

# ** Use this for LWA
# antenna_file = '/home/beards/inst_config/LWA_antenna_locs.txt'

identical_antennas = True
ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0, 1, 2, 3))
ant_info[:, 1] -= NP.mean(ant_info[:, 1])
ant_info[:, 2] -= NP.mean(ant_info[:, 2])
# ant_info[:,3] -= NP.mean(ant_info[:,3])
ant_info[:, 3] = 0.0

core_ind = NP.logical_and((NP.abs(ant_info[:, 1]) < 150.0), (NP.abs(ant_info[:, 2]) < 150.0))
ant_info = ant_info[core_ind, :]

n_antennas = ant_info.shape[0]

# Setup beam
ant_sizex = 4.4  # meters
ant_sizey = 4.4

ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1', 'P2']}
ant_kernshape = {pol: 'rect' for pol in ['P1', 'P2']}
ant_lookupinfo = None

ant_kernshapeparms = {pol: {'xmax': 0.5 * ant_sizex, 'ymax': 0.5 * ant_sizey,
                            'rmin': 0.0, 'rmax': 0.5 * NP.sqrt(ant_sizex**2 + ant_sizey**2),
                            'rotangle': 0.0} for pol in ['P1', 'P2']}

ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype,
                         shape=ant_kernshape, parms=ant_kernshapeparms,
                         lkpinfo=ant_lookupinfo, load_lookup=True)
if identical_antennas:
    ant_aprtrs = [ant_aprtr] * n_antennas

# set up antenna array
ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i, 0])), lat, ant_info[i, 1:],
                     f0, nsamples=nts, aperture=ant_aprtrs[i])
    ant.f = ant.f0 + DSP.spectax(2 * nts, dt, shift=True)
    ants += [ant]
    aar = aar + ant

aar.grid(xypad=2 * NP.max([ant_sizex, ant_sizey]))

antpos_info = aar.antenna_positions(sort=True)

# *** Set up sky model

src_seed = 50  # fix the seed
rstate = NP.random.RandomState(src_seed)
NP.random.seed(src_seed)

n_src = 10
# lmrad = rstate.uniform(low=0.0,high=0.2,size=n_src).reshape(-1,1)
lmrad = rstate.uniform(low=0.0, high=0.05, size=n_src).reshape(-1, 1)**(0.5)  # uniform distributed
lmrad[-1] = 0.05
lmang = rstate.uniform(low=0.0, high=2 * NP.pi, size=n_src).reshape(-1, 1)
skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang))).reshape(-1, 2)
skypos = NP.hstack((skypos, NP.sqrt(1.0 - (skypos[:, 0]**2 + skypos[:, 1]**2)).reshape(-1, 1)))
src_flux = NP.sort((NP.random.uniform(low=0.3, high=0.7, size=n_src)))
src_flux[-1] = 1.0
# src_flux = 10.0*NP.ones(n_src)
# src_flux = 10.0*NP.array([.853,.968,.959,.906,.916,.511,.903,.671,.691,.941])
# src_flux = 3.0*NP.array([.853,4*.968,.959,.906,.916,.511,.903,.671,.691,.941])

tot_flux = NP.sum(src_flux)
frac_flux = 0.0
ind = 0
while frac_flux < model_frac:
    ind += 1
    frac_flux = NP.sum(src_flux[-ind:]) / tot_flux

sky_model = NP.zeros((n_src, nchan, 4))
sky_model[:, :, 0:3] = skypos.reshape(n_src, 1, 3)
sky_model[:, :, 3] = src_flux.reshape(n_src, 1)
# sky_model=sky_model[0,:,:].reshape(1,nchan,4)
sky_model = sky_model[-ind:, :, :]

# ***  set up calibration
calarr = {}
ant_pos = ant_info[:, 1:]  # I'll let the cal class put it in wavelengths.

# auto_noise_model = rxr_noise

for pol in ['P1', 'P2']:
    calarr[pol] = EPICal.cal(freqs, ant_pos, pol=pol, sim_mode=True, n_iter=cal_iter,
                             damping_factor=0.35, inv_gains=False, sky_model=sky_model,
                             exclude_autos=True)

# Create array of gains to watch them change
ncal = itr / cal_iter
cali = 0
gain_stack = NP.zeros((ncal + 1, ant_info.shape[0], nchan), dtype=NP.complex64)

for i in xrange(itr):
    print i
    # simulate
    E_timeseries_dict = SIM.stochastic_E_timeseries(f_center, nchan / 2, 2 * channel_width,
                                                    flux_ref=src_flux, skypos=skypos,
                                                    antpos=antpos_info['positions'],
                                                    tshift=False)

    timestamp = str(DT.datetime.now())
    update_info = {}
    update_info['antennas'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp
    for label in aar.antennas:
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        ind = antpos_info['labels'].index(label)
        adict['t'] = E_timeseries_dict['t']
        adict['gridfunc_freq'] = 'scale'
        adict['gridmethod'] = 'NN'
        adict['distNN'] = 3.0
        adict['Et'] = {}
        adict['flags'] = {}
        adict['wtsinfo'] = {}
        for pol in ['P1', 'P2']:
            adict['flags'][pol] = False
            adict['Et'][pol] = E_timeseries_dict['Et'][:, ind]
            adict['wtsinfo'][pol] = [{'orientation': 0.0,
                                      'lookup': '/data3/t_nithyanandan/'
                                      'project_MOFF/simulated/LWA/data/lookup/'
                                      'E_illumination_isotropic_radiators_lookup_zenith.txt'}]

        update_info['antennas'] += [adict]

    aar.update(update_info, parallel=False, verbose=False, nproc=16)

    # Calibration steps
    # read in data array
    aar.caldata['P1'] = aar.get_E_fields('P1', sort=True)
    ideal_data = aar.caldata['P1']['E-fields'][0, :, :].copy()
    # tempdata[:,2]/=NP.abs(tempdata[0,2]) # uncomment this line to make noise = 0 for single source
    tempdata = calarr['P1'].apply_cal(ideal_data, meas=True)
    tempdata += (NP.sqrt(rxr_noise) / NP.sqrt(2) *
                 (NP.random.normal(loc=0.0, scale=1, size=tempdata.shape) + 1j *
                  NP.random.normal(loc=0.0, scale=1, size=tempdata.shape)))
    # Apply calibration and put back into antenna array
    aar.caldata['P1']['E-fields'][0, :, :] = calarr['P1'].apply_cal(tempdata)

    if grid_map_method == 'regular':
        aar.grid_convolve_new(pol='P1', method='NN', distNN=0.5 * FCNST.c / f0,
                              tol=1.0e-6, maxmatch=1, identical_antennas=True,
                              gridfunc_freq='scale', mapping='weighted',
                              wts_change=False, parallel=False, pp_method='queue',
                              nproc=16, cal_loop=True, verbose=False)
    else:
        if i == 0:
            aar.genMappingMatrix(pol='P1', method='NN', distNN=0.5 *
                                 NP.sqrt(ant_sizex**2 + ant_sizey**2) /
                                 NP.sqrt(2), identical_antennas=True,
                                 gridfunc_freq='scale', wts_change=False,
                                 parallel=False)

    if i == 0:
        imgobj = AA.NewImage(antenna_array=aar, pol='P1')
    else:
        imgobj.update(antenna_array=aar, reset=True)

    imgobj.imagr(weighting='natural', pol='P1', pad=0, verbose=False,
                 grid_map_method=grid_map_method, cal_loop=True, stack=False)

    if (make_ideal_cal & (cali == (cal_iter - 1))):
        aar.caldata['P1']['E-fields'][0, :, :] = ideal_data
        if i == 0:
            imgobj_ideal = AA.NewImage(antenna_array=aar, pol='P1')
        else:
            imgobj_ideal.update(antenna_array=aar, reset=True)

        imgobj_ideal.imagr(weighting='natural', pol='P1', pad=0, verbose=False,
                           grid_map_method=grid_map_method, cal_loop=True, stack=False)

    # update calibration
    calarr['P1'].update_cal(tempdata, imgobj)

    if i == 0:
        avg_img = imgobj.img['P1'].copy()
        im_stack = NP.zeros((ncal, avg_img.shape[0], avg_img.shape[1]), dtype=NP.double)
        im_stack[cali, :, :] = avg_img[:, :, 2].copy()
        temp_im = avg_img[:, :, 2]
        if make_ideal_cal:
            avg_img_ideal = NP.zeros((ncal, avg_img.shape[0], avg_img.shape[1]), dtype=NP.double)

        gain_stack[cali, :, :] = calarr['P1'].curr_gains
        cali += 1

    else:
        avg_img = avg_img + imgobj.img['P1'].copy()
        temp_im = temp_im + imgobj.img['P1'][:, :, 2].copy()
        if (make_ideal_cal & (cali == (cal_iter - 1))):
            avg_img_ideal = avg_img_ideal + imgobj_ideal.img['P1'][:, :, 2].copy() / cal_iter

        if i % cal_iter == 0:
            im_stack[cali, :, :] = temp_im / cal_iter
            temp_im[:] = 0.0
            gain_stack[cali, :, :] = calarr['P1'].curr_gains
            cali += 1

    if True in NP.isnan(calarr['P1'].cal_corr):
        print 'NAN in calibration gains! exiting!'
        break

    avg_img /= itr

t2 = time.time()

print 'Full loop took ', t2 - t1, 'seconds'

# *** Do some plotting
# TODO: change to object oriented plotting

pre_im = im_stack[1, :, :]
f_pre_im = PLT.figure("pre_im")
clf()
imshow(pre_im, aspect='equal', origin='lower',
       extent=(imgobj.gridl.min(), imgobj.gridl.max(),
               imgobj.gridm.min(), imgobj.gridm.max()),
       interpolation='none')
xlim([-.3, .3])
ylim([-.3, .3])
# clim([0.0*NP.nanmin(pre_im),0.5*NP.nanmax(pre_im)])
xlabel('l')
ylabel('m')
# cb=colorbar(orientation='horizontal',ticks=[0,1200,2400,3600],pad=.1,shrink=.6,label='Jy/beam')
title('Before Calibration')

post_im = im_stack[-2, :, :]
f_post_im = PLT.figure("post_im")
clf()
imshow(post_im, aspect='equal', origin='lower',
       extent=(imgobj.gridl.min(), imgobj.gridl.max(),
               imgobj.gridm.min(), imgobj.gridm.max()),
       interpolation='none')
xlim([-.3, .3])
ylim([-.3, .3])
# clim([0.0*NP.nanmin(post_im),0.5*NP.nanmax(post_im)])
xlabel('l')
ylabel('m')
# cb=colorbar(orientation='horizontal',ticks=[0,2500,5000,7500],pad=.1,shrink=.6,label='Jy/beam')
plot(sky_model[:, 0, 0], sky_model[:, 0, 1], 'o', mfc='none', mec='red', mew=1, ms=10)
title('After Calibration')

if make_ideal_cal:
    ideal_im = avg_img_ideal
    f_ideal_im = PLT.figure("ideal_im")
    clf()
    imshow(ideal_im, aspect='equal', origin='lower',
           extent=(imgobj_ideal.gridl.min(), imgobj_ideal.gridl.max(),
                   imgobj_ideal.gridm.min(), imgobj_ideal.gridm.max()),
           interpolation='none')
    xlim([-.3, .3])
    ylim([-.3, .3])
    # clim([0.0*NP.nanmin(post_im),0.5*NP.nanmax(post_im)])
    xlabel('l')
    ylabel('m')
    # cb=colorbar(orientation='horizontal',ticks=[0,2500,5000,7500],pad=.1,shrink=.6,label='Jy/beam')
    plot(sky_model[:, 0, 0], sky_model[:, 0, 1], 'o', mfc='none', mec='red', mew=1, ms=10)
    title('Perfect Calibration')

# remove some arbitrary phases.
data = gain_stack[0:-1, :, 2] * (calarr['P1'].sim_gains[calarr['P1'].ref_ant, 2] *
                                 NP.conj(gain_stack[1, calarr['P1'].ref_ant, 2]) /
                                 NP.abs(calarr['P1'].sim_gains[calarr['P1'].ref_ant, 2] *
                                 gain_stack[1, calarr['P1'].ref_ant, 2]))
true_g = calarr['P1'].sim_gains[:, 2]

# Phase and amplitude convergence
f_phases = PLT.figure("Phases")
f_amps = PLT.figure("Amplitudes")
for i in xrange(gain_stack.shape[1]):
    PLT.figure(f_phases.number)
    plot(NP.angle(data[:, i] * NP.conj(true_g[i])))
    PLT.figure(f_amps.number)
    plot(NP.abs(data[:, i] / true_g[i]))
PLT.figure(f_phases.number)
xlim([0, 20])
ylim([-NP.pi, NP.pi])
xlabel('Calibration Iteration')
ylabel('Phase error (rad)')
PLT.figure(f_amps.number)
xlim([0, 20])
xlabel('Calibration Iteration')
ylabel('Relative amplitude')

# Histogram
# f_hist = PLT.figure("Histogram")
# PLT.hist(NP.real(data[-1,:]-true_g),histtype='step')
# PLT.hist(NP.imag(data[-1,:]-true_g),histtype='step')

# Expected noise
# Nmeas_eff = itr
# Nmeas_eff = 100
Nmeas_eff = cal_iter / (calarr['P1'].damping_factor)
visvar = NP.sum(sky_model[:, 2, 3])**2 / Nmeas_eff
gvar = 4 * visvar / (NP.sum(abs(true_g.reshape(1, calarr['P1'].n_ant) *
                                calarr['P1'].model_vis[:, :, 2])**2, axis=1) -
                     NP.abs(true_g * NP.diag(calarr['P1'].model_vis[:, :, 2])))


# with open('/data2/beards/tmp/sim_run.pickle','w') as f:
#    pickle.dump([calarr,gain_stack,im_stack,sky_model,cal_iter],f)

# Get SNR on each source
region_size = 0.1  # units of l
psf_size = .02
bg_map = post_im.copy()
for i in np.arange(n_src):
    ind = np.where((np.sqrt((imgobj.gridl - sky_model[i, 0, 0])**2 +
                            (imgobj.gridm - sky_model[i, 0, 1])**2) < psf_size))
    bg_map[ind] = NP.nan

source_snr = np.zeros(n_src)
source_s = np.zeros(n_src)
nbins = 100
for i in np.arange(n_src):
    ind1 = np.where((np.sqrt((imgobj.gridl - sky_model[i, 0, 0])**2 +
                             (imgobj.gridm - sky_model[i, 0, 1])**2) < psf_size))
    ind2 = np.where((np.sqrt((imgobj.gridl - sky_model[i, 0, 0])**2 +
                             (imgobj.gridm - sky_model[i, 0, 1])**2) < region_size))
    source_ind = NP.unravel_index(NP.argmin((imgobj.gridl - sky_model[i, 0, 0])**2 +
                                  (imgobj.gridm - sky_model[i, 0, 1])**2), post_im.shape)
    # source_snr[i] = (NP.nanmax(post_im[ind1]) - NP.nanmean(bg_map[ind2])) / NP.nanstd(bg_map[ind2])
    source_snr[i] = (post_im[source_ind] - NP.nanmean(bg_map[ind2])) / NP.nanstd(bg_map[ind2])
    source_s[i] = post_im[source_ind]

snr_map = np.zeros(post_im.shape)
noise_map = np.zeros(post_im.shape)
signal_map = np.zeros(post_im.shape)
for i in np.arange(post_im.shape[0]):
    m = imgobj.gridm[i, 0]
    print m
    if abs(m) > .3:
        continue
    for j in np.arange(post_im.shape[1]):
        l = imgobj.gridl[i, j]
        if (sqrt(l**2 + m**2) > .3):
            continue
        ind1 = np.where((np.sqrt((imgobj.gridl - l)**2 + (imgobj.gridm - m)**2) < psf_size))
        ind2 = np.where((np.sqrt((imgobj.gridl - l)**2 + (imgobj.gridm - m)**2) < region_size))

        # signal_map[i, j] = NP.abs(post_im[i, j] - NP.nanmean(bg_map[ind2]))
        signal_map[i, j] = post_im[i, j] - NP.nanmean(bg_map[ind2])
        noise_map[i, j] = NP.nanstd(bg_map[ind2])
        # noise_map[i, j] = NP.nanmedian(NP.abs(bg_map[ind2] - NP.nanmedian(bg_map[ind2])))
