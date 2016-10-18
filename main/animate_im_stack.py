import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

# DRAW A FIGURE WITH MATPLOTLIB

fps = 5.0
duration = (im_stack.shape[0] - 2.0) / fps

fig_mpl, ax = plt.subplots(1, figsize=(10, 10), facecolor='white')
ax.set_title("LWA-OV observation")
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)

# Make first frame
img = imshow(im_stack[1, :, :], aspect='equal', extent=(imgobj.gridl.min(),
                                                        imgobj.gridl.max(),
                                                        imgobj.gridm.min(),
                                                        imgobj.gridm.max()),
             interpolation='none', origin='lower')
img.set_clim([np.nanmin(avg_img_no_autos), np.nanmax(avg_img_no_autos)])


def make_frame_mpl(t):
    # Update data
    img.set_data(im_stack[fps * t + 1, :, :])
    ax.set_title('t = {:06.3f} (ms)'.format(1e3 * t * fps * dT * cal_iter))
    return mplfig_to_npimage(fig_mpl)  # RGB image of the figure

animation = mpy.VideoClip(make_frame_mpl, duration=duration)
animation.write_gif("~/test.gif", fps=fps)
