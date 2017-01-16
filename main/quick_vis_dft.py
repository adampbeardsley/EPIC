# A quick script to do a DFT of visibility data

import numpy as np


def quick_vis_dft(visdata, u, v, gridl, gridm):
    # Assume visdata is nant x nant, uv is nant x nant x 2, and gridl is same as gridm
    image = np.zeros_like(gridl)
    visdata = visdata.reshape(-1)
    u = u.reshape(-1)
    v = v.reshape(-1)
    for i in arange(visdata.size):
        image += np.real(visdata[i] * np.exp(2 * np.pi * 1j * (u[i] * gridl + v[i] * gridm)))

    return image
