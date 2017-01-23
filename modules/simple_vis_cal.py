# Simple script to do basic visiblity-based calibration
import numpy as NP


def vis_cal(visdata, vismodel, max_iter=2000, threshold=0.000001,
            damping_factor=0.05, ref_ant=0, inv_gains=False, uv=None, min_u=None):

    nchan = visdata.shape[-1]
    nant = visdata.shape[0]
    visdata = visdata.reshape(nant**2, nchan)
    vismodel = vismodel.reshape(nant**2, nchan)
    gains = NP.ones((nant, nchan), dtype=NP.complex64)

    # set up bookkeeping
    ant1 = NP.arange(nant)
    chi_history = NP.zeros((max_iter, nchan))

    A = NP.zeros((nant**2, nant), dtype=NP.complex64)  # matrix for minimization
    ind2 = NP.repeat(NP.arange(nant), nant)
    ind3 = NP.tile(NP.arange(nant), nant)

    if ((uv is not None) and (min_u is not None)):
        # Trim based on baseline length
        u_mag = NP.sqrt(NP.sum(uv**2, axis=-1))
        u_mag = u_mag.reshape(-1)
        short_ind = NP.where(u_mag < min_u)
        ind2 = NP.delete(ind2, short_ind)
        ind3 = NP.delete(ind3, short_ind)
        visdata = NP.delete(visdata, short_ind, axis=0)
        vismodel = NP.delete(vismodel, short_ind, axis=0)
        A = NP.delete(A, short_ind, axis=0)

    auto_ind = NP.where(ind2 == ind3)
    ind2 = NP.delete(ind2, auto_ind)
    ind3 = NP.delete(ind3, auto_ind)
    visdata = NP.delete(visdata, auto_ind, axis=0)
    vismodel = NP.delete(vismodel, auto_ind, axis=0)
    A = NP.delete(A, auto_ind, axis=0)
    ind1 = NP.arange(A.shape[0])

    for fi in xrange(nchan):
        tempgains = gains[:, fi].copy()
        tempdata = visdata[:, fi]
        tempmodel = vismodel[:, fi]
        tries = 0
        change = 100.0
        while (tries < max_iter) and (change > threshold):
            prevgains = tempgains.copy()
            if inv_gains:
                chi_history[tries, fi] = NP.sum(NP.abs(tempmodel - tempgains[ind2] *
                                                       NP.conj(tempgains[ind3]) * tempdata)**2)
                A[ind1, ind2] = tempdata * NP.conj(prevgains[ind3])
                tempgains = NP.linalg.lstsq(A, tempmodel)[0]
            else:
                chi_history[tries, fi] = NP.sum(NP.abs(tempdata - tempgains[ind2] *
                                                       NP.conj(tempgains[ind3]) * tempmodel)**2)
                A[ind1, ind2] = tempmodel * NP.conj(prevgains[ind3])
                tempgains = NP.linalg.lstsq(A, tempdata)[0]
            tempgains = tempgains * NP.conj(tempgains[ref_ant] / NP.abs(tempgains[ref_ant]))
            tempgains = (1 - damping_factor) * tempgains + damping_factor * prevgains
            change = NP.max(NP.abs(tempgains - prevgains) / NP.abs(prevgains))
            tries += 1
        if tries == max_iter:
            print 'Warning! Vis calibration failed to converge. Continuing'
        gains[:, fi] = tempgains.copy()

    return gains
