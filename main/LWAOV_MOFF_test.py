import datetime as DT
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import scipy.constants as FCNST
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import progressbar as PGB
import antenna_array as AA
import data_interface as DI
import geometry as GEOM
import sim_observe as SIM
import my_DSP_modules as DSP
from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
import ipdb as PDB

infile = '/data5/LWA_OV_data/data_raw/jun11/47mhz/2016-06-11-08:00:37_0000032252559360.000000.dada.CDF'
du = DI.DataHandler(indata=infile)
max_n_timestamps = 4

lat = du.latitude
f0 = du.center_freq
nchan = du.nchan
nts = nchan / 2  # For now this is necessary to get the aar to create freq-ordered data
fs = du.sample_rate
dt = 1 / fs
freqs = du.freq
channel_width = du.freq_resolution
f_center = f0
bchan = 0
echan = 108
max_antenna_radius = 100.0  # in meters - Should pick up most of core
antid = du.antid
antpos = du.antpos
n_antennas = du.n_antennas
timestamps = du.timestamps
n_timestamps = du.n_timestamps
npol = du.npol
ant_data = du.data[0:max_n_timestamps, :, :, :]

core_ind = NP.logical_and((NP.abs(antpos[:, 0]) < max_antenna_radius),
                          (NP.abs(antpos[:, 1]) < max_antenna_radius))
antid = antid[core_ind]
antpos = antpos[core_ind, :]
ant_info = NP.hstack((antid.reshape(-1, 1), antpos))
n_antennas = ant_info.shape[0]
ant_data = ant_data[:, core_ind, :, :]

ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i, 0])), lat, ant_info[i, 1:], f0, nsamples=nts)
    ant.f = ant.f0 + DSP.spectax(nchan, dt, shift=True)
    ants += [ant]
    aar = aar + ant

aar.grid()

antpos_info = aar.antenna_positions(sort=True)

if max_n_timestamps is None:
    max_n_timestamps = len(timestamps)
else:
    max_n_timestamps = min(max_n_timestamps, len(timestamps))

timestamps = timestamps[:max_n_timestamps]

# stand_cable_delays = NP.loadtxt('/data3/t_nithyanandan/project_MOFF/data/samples/cable_delays.txt', skiprows=1)
# antennas = stand_cable_delays[:,0].astype(NP.int).astype(str)
# cable_delays = stand_cable_delays[:,1]
cable_delays = np.zeros_like(antpos)  # Need to work this out from cal

for it in xrange(max_n_timestamps):
    timestamp = timestamps[it]
    update_info = {}
    update_info['antennas'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp

    print 'Consolidating Antenna updates...'
    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-',
                                        left=' |', right='| '), PGB.Counter(),
                                        '/{0:0d} Antennas '.format(n_antennas),
                                        PGB.ETA()],
                               maxval=n_antennas).start()
    antnum = 0
    for ia, label in enumerate(antid):
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        adict['t'] = NP.arange(nts) * dt
        adict['gridfunc_freq'] = 'scale'
        adict['gridmethod'] = 'NN'
        adict['distNN'] = 0.5 * FCNST.c / f0
        adict['tol'] = 1.0e-6
        adict['maxmatch'] = 1
        adict['Et'] = {}
        adict['flags'] = {}
        adict['stack'] = True
        adict['wtsinfo'] = {}
        adict['delaydict'] = {}
        for ip in range(npol):
            adict['delaydict']['P{0}'.format(ip + 1)] = {}
            adict['delaydict']['P{0}'.format(ip + 1)]['frequencies'] = freqs
            # adict['delaydict']['P{0}'.format(ip + 1)]['delays'] = cable_delays[antennas == label]
            adict['delaydict']['P{0}'.format(ip + 1)]['fftshifted'] = True
            adict['wtsinfo']['P{0}'.format(ip + 1)] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            adict['Et']['P{0}'.format(ip + 1)] = ant_data[it, ia, :nts, ip]  # Stuff with the wrong data
            if NP.any(NP.isnan(adict['Et']['P{0}'.format(ip + 1)])):
                adict['flags']['P{0}'.format(ip + 1)] = True
            else:
                adict['flags']['P{0}'.format(ip + 1)] = False

        update_info['antennas'] += [adict]

        progress.update(antnum + 1)
        antnum += 1
    progress.finish()

    aar.update(update_info, parallel=True, verbose=True)
    aar.caldata['P1'] = aar.get_E_fields('P1')  # Trick aar into thinking it's doing the right thing.
    aar.caldata['P1']['E-fields'][0, :, :] = ant_data[it, :, :]  # Put real data in
    aar.grid_convolve(pol='P1', method='NN', distNN=0.5 * FCNST.c / f0,
                      tol=1.0e-6, maxmatch=1, identical_antennas=True,
                      cal_loop=False, gridfunc_freq='scale', mapping='weighted',
                      wts_change=False, parallel=True, pp_method='pool')

    # fp1 = [ad['flags']['P1'] for ad in update_info['antennas']]
    # p1f = [a.antpol.flag['P1'] for a in aar.antennas.itervalues()]
    imgobj = AA.NewImage(antenna_array=aar, pol='P1')
    imgobj.imagr(weighting='natural', pol='P1')
    img = imgobj.img['P1']

    # for chan in xrange(imgobj.holograph_P1.shape[2]):
    #     imval = NP.abs(imgobj.holograph_P1[imgobj.mf_P1.shape[0]/2,:,chan])**2 # a horizontal slice
    #     imval = imval[NP.logical_not(NP.isnan(imval))]
    #     immax2[it,chan,:] = NP.sort(imval)[-2:]

    if it == 0:
        avg_img = NP.copy(img)
    else:
        avg_img += NP.copy(img)
    if NP.any(NP.isnan(avg_img)):
        PDB.set_trace()

avg_img /= max_n_timestamps

beam = imgobj.beam['P1']

fig = PLT.figure()
ax = fig.add_subplot(111)
imgplot = ax.imshow(NP.mean(avg_img[:,:,bchan:echan+1], axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())
ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/MOFF_image_{0:0d}_iterations.png'.format(max_n_timestamps), bbox_inches=0)

fig = PLT.figure()
ax = fig.add_subplot(111)
imgplot = ax.imshow(NP.mean(beam[:,:,bchan:echan+1], axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())
ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/MOFF_psf_square_illumination.png'.format(max_n_timestamps), bbox_inches=0)
