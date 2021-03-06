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
from astroutils import geometry as GEOM
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
import h5py

t1 = time.time()

# Make some choices about the analysis
max_n_timestamps = 10
min_timestamp = 0
bchan = 0  # beginning channel (to cut out edges of the bandpass)
echan = 108  # ending channel
max_antenna_radius = 400.0  # meters. To cut outtrigger(s)
pols_use = ['P1']
npol_use = len(pols_use)
apply_cal = True
MOFF_tbinsize = None

# Use absolute antenna locs to get lat and lon
antfile = '/data5/LWA_OV_data/m_files/antenna-positions.txt'
# Location array is 256x3, in ITRF coordinates
ant_locs_xyz = NP.loadtxt(antfile, delimiter=',')
xyz_ref = NP.mean(ant_locs_xyz, axis=0, keepdims=True)
lat0, lon0, alt0 = GEOM.ecef2lla(xyz_ref[:, 0], xyz_ref[:, 1], xyz_ref[:, 2], units='radians')
lat0 = NP.degrees(lat0)
lon0 = NP.degrees(lon0)

# Load in cal file, which is also used for flagging
calfile = '/data5/LWA_OV_data/m_files/flagged-calibration.mat'
cal = sio.loadmat(calfile)
gains = cal['gains'] / np.sqrt(np.mean(np.abs(cal['gains'])**2.))  # Not sure why this is done
calflags = cal['flags']
flags = set(NP.where(NP.any(calflags, axis=1))[0])  # This particular set of flags is all or none per antenna

# Get file, read in header
datafile = '/data5/LWA_OV_data/data_reformatted/jun11/47mhz/2016-06-11-08-00-37_0000000000000000.000000.dada.hdf5'
with h5py.File(datafile, 'r') as fileobj:
    ntimes = fileobj['header']['ntimes'].value  # Number of time samples
    nant = fileobj['header']['nant'].value  # Numer of antennas
    nchan = fileobj['header']['nchan'].value  # Number of frequency channels
    npol = fileobj['header']['npol'].value  # Number of polarizations
    f0 = fileobj['header']['f0'].value  # Center frequency (Hz)
    freq_resolution = fileobj['header']['df'].value  # (Hz)
    bw = fileobj['header']['bw'].value  # Bandwidth (Hz)
    dT = fileobj['header']['dT'].value  # Sampling period (1/df, in s)
    dts = fileobj['header']['dts'].value  # Sampling rate (1/B, in s)
    channels = fileobj['spectral_info']['f'].value  # Frequency channels (Hz)
    antpos = fileobj['antenna_parms']['antpos'].value  # Antenna positions, E-N-alt, in local meters
    antid = fileobj['antenna_parms']['ant_id'].value  # Labels for antennas
antpos[:, 0] = antpos[:, 0] - np.mean(antpos[:, 0]) - 50  # readjust center of array
antpos[:, 1] = antpos[:, 1] - np.mean(antpos[:, 1]) + 50

# Flag antennas outside region of interest
outtriggerind = set(NP.where((NP.abs(antpos[:, 0]) > max_antenna_radius) |
                             (NP.abs(antpos[:, 1]) > max_antenna_radius))[0])
flags.update(outtriggerind)  # Add to set of flags

# ### Antenna and array initialization
ant_info = NP.hstack((antid.reshape(-1, 1), antpos))
n_antennas = ant_info.shape[0]

# Set up the beam
grid_map_method = 'sparse'
identical_antennas = True
ant_sizex = 3.0  # meters
ant_sizey = 3.0
ant_diameter = NP.sqrt(ant_sizex**2 + ant_sizey**2)
ant_kernshape = {pol: 'rect' for pol in pols_use}
ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in pols_use}
ant_lookupinfo = None

ant_kernshapeparms = {pol: {'xmax': 0.5 * ant_sizex, 'ymax': 0.5 * ant_sizey,
                            'rmin': 0.0, 'rmax': 0.5 * ant_diameter,
                            'rotangle': 0.0} for pol in pols_use}
ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype,
                         shape=ant_kernshape, parms=ant_kernshapeparms,
                         lkpinfo=ant_lookupinfo, load_lookup=True)
if identical_antennas:
    ant_aprtrs = [ant_aprtr] * n_antennas

# Set up antenna array
ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    if i in flags:
        continue
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i, 0])), '0', lat0, lon0, ant_info[i, 1:],
                     f0, nsamples=nchan, aperture=ant_aprtrs[i])
    ant.f = channels
    ants += [ant]
    aar = aar + ant

aar.grid(uvspacing=0.4, xypad=2 * NP.max([ant_sizex, ant_sizey]))
antpos_info = aar.antenna_positions(sort=True, centering=True)

# Select time steps
if max_n_timestamps is None:
    max_n_timestamps = ntimes
else:
    max_n_timestamps = min(max_n_timestamps, ntimes)

dstream = DI.DataStreamer()

master_pb = PGB.ProgressBar(widgets=[PGB.Percentage(),
                                     PGB.Bar(marker='-', left=' |', right='| '),
                                     PGB.Counter(),
                                     '/{0:0d} time stamps '.format(max_n_timestamps),
                                     PGB.ETA()], maxval=max_n_timestamps).start()

for i in xrange(max_n_timestamps):

    timestamp = i * dT
    update_info = {}
    update_info['antennas'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp

    dstream.load(datafile, i, datatype='Ef', pol=None)
    print 'Consolidating Antenna updates...'
    antnum = 0
    for label in aar.antennas:
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
        for ip, pol in enumerate(pols_use):
            adict['flags'][pol] = False
            adict['wtsinfo'][pol] = [{'orientation': 0.0, 'lookup': '/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            adict['Ef'][pol] = (dstream.data[pol]['real'][label, :].astype(NP.float32) +
                                1j * dstream.data[pol]['imag'][label, :].astype(NP.float32))
            if apply_cal:
                adict['Ef'][pol] *= gains[ip, label, :].astype(NP.complex64)

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

    master_pb.update(i + 1)

avg_img /= max_n_timestamps
master_pb.finish()

t2 = time.time()

print 'Full loop took ', t2 - t1, 'seconds'

avg_uv = NP.fft.fftshift(NP.fft.fft2(avg_img))
avg_uv[255:258, 255:258] = 0
# avg_uv[127:130, 127:130] = 0
avg_img_no_autos = NP.real(NP.fft.ifft2(NP.fft.fftshift(avg_uv)))

nanind = NP.where(imgobj.gridl**2 + imgobj.gridm**2 > 1.0)
avg_img_no_autos[nanind] = NP.nan  # mask out non-physical pixels

f_image = PLT.figure("LWA OV Image")
clf()
imshow(avg_img_no_autos, aspect='equal',  # origin='lower',
       extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()),
       interpolation='none')
xlim([-1.0, 1.0])
ylim([-1.0, 1.0])
xlabel('l')
ylabel('m')
colorbar()
