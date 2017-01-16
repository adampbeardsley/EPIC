# Simple script to do basic visiblity-based calibration
import numpy as NP


def vis_cal(visdata, vismodel, max_iter=2000, threshold=0.0001, inv_gains=False):

    nchan = visdata.shape[-1]
    nant = visdata.shape[0]
    gains = NP.ones((nant, nchan), dtype=NP.complex64)

    # set up bookkeeping
    ant1 = NP.arange(nant)

    chi_history = NP.zeros((max_iter, nchan))

    tries = 0.0
    change = 100.0
    A = NP.zeros((nant**2, nant), dtype=NP.complex64)  # matrix for minimization
    ind1 = NP.arange(nant**2)
    ind2 = NP.repeat(NP.arange(nant), nant)
    ind3 = NP.tile(NP.arange(nant), nant)
    for fi in xrange(nchan):
        tempgains = gains[:, fi].copy()
        tempdata = visdata[:, :, fi].reshape(-1)
        tempmodel = vismodel[:, :, fi].reshape(-1)
        while (tries < max_iter) and (change > threshold):
            prevgains = tempgains.copy()
            if inv_gains:
                chi_history[tries, fi] = NP.sum(NP.abs(NP.outer(tempgains, NP.conj(tempgains)) *
                                                       tempdata - tempmodel)**2)
                A[ind1, ind2] = tempdata * NP.conj(prevgains[ind3])
                tempgains = NP.linalg.lstsq(A, tempmodel)[0]
            else:
                chi_history[tries, fi] = NP.sum(NP.abs(tempdata - NP.outer(tempgains, NP.conj(tempgains)) *
                                                       tempmodel)**2)
                A[ind1, ind2] = tempmodel * NP.conj(prevgains[ind3])
                tempgains = NP.linalg.lstsq(A, tempdata)[0]
            change = NP.median(NP.abs(tempgains - prevgains) / NP.abs(prevgains))
            tries += 1
        if tries == max_iter:
            print 'Warning! Vis calibration failed to converge. Continuing'
        gains[:, fi] = tempgains.copy()

    return gains
