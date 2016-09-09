import datetime as DT
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import scipy.constants as FCNST
import scipy.io as sio
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import progressbar as PGB
import antenna_array as AA
import data_interface as DI
import geometry as GEOM
import sim_observe as SIM
import my_DSP_modules as DSP
import ipdb as PDB
import EPICal
import aperture as APR
import time
from pygsm import GlobalSkyModel
import healpy as HP
import aipy
from astropy.coordinates import Galactic, FK5
from astropy import units
import astropy.time as AT
import ipdb as PDB
import pickle
from scipy import interpolate

t1 = time.time()

# Get file, read in header
infile = '/data5/LWA_OV_data/data_raw/jun11/47mhz/2016-06-11-08:00:37_0000000000000000.000000.dada.CDF'
du = DI.DataHandler(indata=infile)
du.antid = NP.sort(du.antid)
lat = du.latitude
f0 = du.center_freq
nchan = du.nchan
nts = nchan / 2
fs = du.sample_rate
dt = 1 / fs
freqs = du.freq
channel_width = du.freq_resolution
f_center = f0
antid = du.antid
antpos = du.antpos.copy()
antpos[:, 0] = antpos[:, 0] - np.mean(antpos[:, 0]) - 50  # readjust center of array
antpos[:, 1] = antpos[:, 1] - np.mean(antpos[:, 1]) + 50
antposraw = antpos.copy()
n_antennas = du.n_antennas
timestamps = du.timestamps
MOFF_tbinsize = None
n_timestamps = du.n_timestamps
npol = du.npol
ant_data = du.data

# Make some choices about the analysis
max_n_timestamps = 10000
min_timestamp = 0
bchan = 0  # beginning channel (to cut out edges of the bandpass)
echan = 108  # ending channel
max_antenna_radius = 100.0  # meters. To cut outtrigger(s)
pols = ['P1']
npol_use = len(pols)
apply_delays = False

apply_cal = True
if apply_cal:
    cal_file = '/data5/LWA_OV_data/m_files/flagged-calibration.mat'
    cal = sio.loadmat(cal_file)
    # I don't know why, but Ryan divides by rms
    gains = cal['gains'] / np.sqrt(np.mean(np.abs(cal['gains'])**2.))
    for pol in arange(npol):
        ant_data[:, :, :, pol] *= gains[pol, :, :].reshape(-1, n_antennas, nchan)

# Option to omit antennas
# This list comes from the flags array from Ryan Monroe
ant_flag = NP.array([0, 58, 72, 74, 75, 76, 77, 78, 82, 87, 90, 91, 92, 93, 104,
                     105, 128, 145, 148, 157, 161, 164, 168, 185, 186, 197, 220,
                     225, 236, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255])
# ant_flag = None

# ### Antenna and array initialization

# Flag antennas by removing them
if ant_flag is not None:
    antid = NP.delete(antid, ant_flag)
    antpos = NP.delete(antpos, ant_flag, axis=0)
    ant_data = NP.delete(ant_data, ant_flag, axis=1)

# Select antennas
core_ind = NP.logical_and((NP.abs(antpos[:, 0]) < max_antenna_radius),
                          (NP.abs(antpos[:, 1]) < max_antenna_radius))
antid = antid[core_ind]
antpos = antpos[core_ind, :]
ant_data = ant_data[:, core_ind, :, :]

ant_info = NP.hstack((antid.reshape(-1, 1), antpos))
n_antennas = ant_info.shape[0]

# Set up the beam
grid_map_method = 'sparse'
identical_antennas = True
ant_sizex = 3.0  # meters
ant_sizey = 3.0
ant_diameter = NP.sqrt(ant_sizex**2 + ant_sizey**2)
ant_kernshape = {pol: 'rect' for pol in ['P1', 'P2']}
ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1', 'P2']}
ant_lookupinfo = None

ant_kernshapeparms = {pol: {'xmax': 0.5 * ant_sizex, 'ymax': 0.5 * ant_sizey,
                            'rmin': 0.0, 'rmax': 0.5 * ant_diameter,
                            'rotangle': 0.0} for pol in ['P1', 'P2']}
ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype,
                         shape=ant_kernshape, parms=ant_kernshapeparms,
                         lkpinfo=ant_lookupinfo, load_lookup=True)
if identical_antennas:
    ant_aprtrs = [ant_aprtr] * n_antennas

# Set up antenna array
ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i, 0])), '0', lat, -118.28, ant_info[i, 1:],
                     f0, nsamples=nchan, aperture=ant_aprtrs[i])
    ant.f = freqs
    ants += [ant]
    aar = aar + ant

aar.grid(uvspacing=0.4, xypad=2 * NP.max([ant_sizex, ant_sizey]))
antpos_info = aar.antenna_positions(sort=True, centering=True)

# Select time steps
if max_n_timestamps is None:
    max_n_timestamps = len(timestamps)
else:
    max_n_timestamps = min(max_n_timestamps, len(timestamps))

timestamps = timestamps[min_timestamp:min_timestamp + max_n_timestamps]

master_pb = PGB.ProgressBar(widgets=[PGB.Percentage(),
                                     PGB.Bar(marker='-', left=' |', right='| '),
                                     PGB.Counter(),
                                     '/{0:0d} time stamps '.format(max_n_timestamps),
                                     PGB.ETA()], maxval=max_n_timestamps).start()

for i in xrange(max_n_timestamps):

    timestamp = timestamps[i]
    update_info = {}
    update_info['antennas'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp

    print 'Consolidating Antenna updates...'
    antnum = 0
    for ia, label in enumerate(antid):
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        adict['gridfunc_freq'] = 'scale'
        adict['gridmethod'] = 'NN'
        adict['distNN'] = 3.0
        adict['tol'] = 1.0e-6
        adict['maxmatch'] = 1
        adict['Ef'] = {}
        adict['flags'] = {}
        adict['stack'] = True
        adict['wtsinfo'] = {}
        if apply_delays:
            adict['delaydict'] = {}
        for ip, pol in enumerate(pols):
            adict['flags'][pol] = False
            adict['wtsinfo'][pol] = [{'orientation': 0.0, 'lookup': '/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            adict['Ef'][pol] = ant_data[i + min_timestamp, ia, :, ip]

        update_info['antennas'] += [adict]

        antnum += 1

    aar.update(update_info, parallel=False, verbose=False)

    if grid_map_method == 'regular':
        aar.grid_convolve_new(pol='P1', method='NN',
                              distNN=0.5 * NP.sqrt(ant_sizex**2 + ant_sizey**2),
                              identical_antennas=False, cal_loop=False,
                              gridfunc_freq='scale', wts_change=False,
                              parallel=False, pp_method='pool')
    else:
        if i == 0:
            aar.genMappingMatrix(pol='P1', method='NN',
                                 distNN=0.5 * NP.sqrt(ant_sizex**2 + ant_sizey**2),
                                 identical_antennas=True, gridfunc_freq='scale',
                                 wts_change=False, parallel=False)

    if i == 0:
        imgobj = AA.NewImage(antenna_array=aar, pol='P1')
    else:
        imgobj.update(antenna_array=aar, reset=True)

    imgobj.imagr(weighting='natural', pol='P1', pad=0, verbose=False,
                 grid_map_method=grid_map_method, cal_loop=False, stack=False)

    if i == 0:
        avg_img = NP.mean(imgobj.img['P1'][:, :, bchan:echan].copy(), axis=2)
    else:
        avg_img = avg_img + NP.mean(imgobj.img['P1'][:, :, bchan:echan].copy(), axis=2)

    avg_img /= max_n_timestamps

    master_pb.update(i + 1)
master_pb.finish()

t2 = time.time()

print 'Full loop took ', t2 - t1, 'seconds'

avg_uv = NP.fft.fftshift(NP.fft.fft2(avg_img))
# avg_uv[253:260, 253:260] = 0
avg_uv[127:130, 127:130] = 0
avg_img_no_autos = NP.real(NP.fft.ifft2(NP.fft.fftshift(avg_uv)))

nanind = NP.where(imgobj.gridl**2 + imgobj.gridm**2 > 1.0)
avg_img_no_autos[nanind] = NP.nan  # mask out non-physical pixels

f_image = PLT.figure("LWA OV Image")
clf()
imshow(avg_img_no_autos, aspect='equal', origin='lower',
       extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()),
       interpolation='none')
xlim([-1.0, 1.0])
ylim([-1.0, 1.0])
xlabel('l')
ylabel('m')
colorbar()
# clim([0.0*NP.nanmin(pre_im),0.5*NP.nanmax(pre_im)])
