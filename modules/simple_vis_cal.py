# Simple script to do basic visiblity-based calibration
import numpy as NP


def vis_cal(visdata, vismodel, max_iter=2000, threshold=0.000001,
            damping_factor=0.05, ref_ant=0, inv_gains=False):

    nchan = visdata.shape[-1]
    nant = visdata.shape[0]
    gains = NP.ones((nant, nchan), dtype=NP.complex64)

    # set up bookkeeping
    ant1 = NP.arange(nant)
    chi_history = NP.zeros((max_iter, nchan))

    A = NP.zeros((nant * (nant - 1), nant), dtype=NP.complex64)  # matrix for minimization
    ind1 = NP.arange(nant * (nant - 1))
    ind2 = NP.repeat(NP.arange(nant), nant)
    ind3 = NP.tile(NP.arange(nant), nant)
    auto_ind = NP.where(ind2 == ind3)
    ind2 = NP.delete(ind2, auto_ind)
    ind3 = NP.delete(ind3, auto_ind)
    for fi in xrange(nchan):
        tempgains = gains[:, fi].copy()
        tempdata = NP.delete(visdata[:, :, fi].reshape(-1), auto_ind)
        tempmodel = NP.delete(vismodel[:, :, fi].reshape(-1), auto_ind)
        tries = 0.0
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
