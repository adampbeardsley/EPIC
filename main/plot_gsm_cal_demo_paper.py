# A script to plot the GSM to match LWA observation
# IMPORTANT: Run this AFTER running the calibrate_LWA_data script - it relies on having certain things already calculated.

import lwa_ant
from datetime import datetime
import scipy.ndimage as ndimage
from pygsm import GlobalSkyModel

#resolution = (FCNST.speed_of_light / f0) / 100.0 # rough resolution of LWA (in radians)

print 'Getting the GSM'
nside = 512
# Load in GSM
gsm = GlobalSkyModel() # Note that I'm not using the 2016 GSM because it excludes certain sources (importantly CygA)
sky = gsm.generate(f0*10**(-6))

# convert sky to Jy/(LWA beam)
Aeff = 100.0**2 * NP.pi /4.0 # basically a filled aperture
sky *= 2 * FCNST.Boltzmann / Aeff * 10**26 # Jy per LWA beam
#sky *= 2*FCNST.Boltzmann / (FCNST.speed_of_light / f0)**2 * 10**26 # Jy/str

#sky = HP.smoothing(sky, fwhm=resolution) # Don't need this anymore because now I'm convolving with PSF

sky = HP.ud_grade(sky,nside) # provided with nside=512, convert to lower res
inds = NP.arange(HP.nside2npix(nside))
theta,phi = HP.pixelfunc.pix2ang(nside,inds)
gc = Galactic(l=phi, b=NP.pi/2-theta, unit=(units.radian,units.radian))
radec = gc.fk5
eq = aipy.coord.radec2eq((-lst+radec.ra.radian,radec.dec.radian))
xyz = NP.dot(aipy.coord.eq2top_m(0,lat*NP.pi/180),eq)

# Keep just pixels above horizon
include = NP.where(xyz[2,:]>0)

# Get beam and gridl,gridm matrices
print 'Retrieving saved beam'
with open('/data2/beards/instr_data/lwa_power_beam.pickle') as f:
    beam,gridl,gridm = pickle.load(f)

beam=beam.flatten()
gridl=gridl.flatten()
gridm=gridm.flatten()
smalll=gridl[0::10]
smallm=gridm[0::10]
smallb=beam[0::10]

print 'Applying beam to GSM'
# attenuate by two factors of the beam
beam_interp = interpolate.griddata((smalll,smallm),smallb,(xyz[0,include],xyz[1,include]),method='linear')
sky[include] = sky[include] * beam_interp**2 # name to match other sky model version

#Rotate sky
lwa1 = lwa_ant.LwaObserver('LWA1')
lwa1.date = datetime(2011,9,21,3,9,00)
lwa1.generate(f0*10**(-6))

# Get RA and DEC of zenith
ra_rad, dec_rad = lwa1.radec_of(0, np.pi/2)
ra_deg  = ra_rad / np.pi * 180
dec_deg = dec_rad / np.pi * 180

# Apply rotation
hrot = HP.Rotator(rot=[ra_deg, dec_deg], coord=['G', 'C'], inv=True)
theta,phi = HP.pix2ang(nside,NP.arange(HP.nside2npix(nside)))
g0, g1 = hrot(theta, phi)
pix0 = HP.ang2pix(nside, g0, g1)
sky_rotated = sky[pix0]

print 'Plotting'

f_gsm = PLT.figure("GSM")
final_map=HP.orthview(sky_rotated,fig=f_gsm.number,half_sky=True,flip='geo',min=0,max=0.5*sky[include].max(),unit='Jy/beam',return_projected_map=True)

# Convolve with PSF of LWA
ind=NP.isinf(final_map)
final_map[ind]=0 # do this so I can FFT it
uv = NP.fft.fftshift(NP.fft.fft2(NP.fft.fftshift(final_map.copy())))

#load previously calculated LWA PSF and convolve
with open('/data2/beards/instr_data/lwa_uv_psf.pickle') as f:
    lwa_uv_psf,gridu,gridv = pickle.load(f)
du = gridu[0,1]-gridu[0,0]
dv = gridv[1,0]-gridv[0,0]
# interpolate to GSM defined grid
gridu_gsm,gridv_gsm = NP.meshgrid(0.5*(NP.arange(uv.shape[1])-uv.shape[1]/2), 0.5*(NP.arange(uv.shape[0])-uv.shape[0]/2))
# map_coordinates uses pixel coordinates... need to remap
ucoords = (gridu_gsm - gridu.min()) / du
vcoords = (gridv_gsm - gridv.min()) / dv
psf_matched = ndimage.map_coordinates(lwa_uv_psf,[ucoords,vcoords])
psf_image = NP.fft.fftshift(NP.real(NP.fft.ifft2(NP.fft.fftshift(psf_matched))))
norm_factor = NP.sum(psf_image[where(abs(psf_image) > 0.05*psf_image.max())]) # normalize the PSF
psf_matched = psf_matched / norm_factor

uv_convolved = psf_matched*uv

final_map = NP.fft.fftshift(NP.real(NP.fft.ifft2(NP.fft.fftshift(uv_convolved))))
final_map[ind] = -inf # reset to -inf to plot it as white

clf()
imshow(final_map,aspect='equal',origin='lower',extent=(-1,1,-1,1),interpolation='none')
xlim([-1.0,1.0])
ylim([-1.0,1.0])
clim([0.0,0.5*NP.nanmax(post_im)]) # Same color scale as final image
xlabel('l')
ylabel('m')
cb=colorbar(orientation='horizontal',ticks=[0,2500,5000,7500],pad=.1,shrink=.6,label='Jy / (LWA beam)')
title('GSM')

