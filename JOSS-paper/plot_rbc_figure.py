import matplotlib.pyplot as plt
import numpy as np
import h5py
import dedalus.extras.plot_tools as plot_tools

plt.style.use('prl')

df0 = h5py.File("snapshots/snapshots_s1.h5","r")
df = h5py.File("snapshots/snapshots_s37.h5","r")

ts = h5py.File("timeseries/timeseries_s1/timeseries_s1_p0.h5","r")


scale = 1
nrows = 1
ncols = 3
image = plot_tools.Box(2, 1)
pad = plot_tools.Frame(0.2, 0.3, 0.4, 0.4)
margin = plot_tools.Frame(0.3, 0.25, 0.1, 0.1)
mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)



skip_x = 8
skip_z = 2

ax2 = mfig.add_axes(0,1,[0,0,1,1])
ax = plot_tools.plot_bot(df['tasks/b'],(1,2), (-1,slice(None),slice(None)),title='buoyancy',axes=ax2)
ax[0].quiver(df['scales/x/1.0/'][::skip_x], df['scales/z/1.0'][::skip_z], df['tasks/u'][-1,::skip_x,::skip_z].T, df['tasks/w'][-1,::skip_x,::skip_z].T)

ax1 = mfig.add_axes(0,0,[0,0,1,1])
ax = plot_tools.plot_bot(df0['tasks/b'],(1,2), (0,slice(None),slice(None)),title='buoyancy',axes=ax1)
ax[0].quiver(df0['scales/x/1.0/'][::skip_x], df0['scales/z/1.0'][::skip_z], df['tasks/u'][0,::skip_x,::skip_z].T, df['tasks/w'][0,::skip_x,::skip_z].T)
ax[1].ticklabel_format(scilimits=(0,0),useMathText=True)
ax[1].text(1.025, 1, r'$\times 10^{-3}$', va='bottom', ha='left',transform=ax[1].transAxes)
ax[1].xaxis.get_children()[1].set_visible(False)


growth = 655.1197879003645
model = ts['tasks/b_rms'][0,0,0]*np.exp(growth*ts['scales/sim_time'][:])
ax3 = mfig.add_axes(0,2,[0,0,1,1])
ax3.semilogy(ts['scales/sim_time'][:],ts['tasks/b_rms'][:,0,0], label='IVP')
ax3.semilogy(ts['scales/sim_time'][:],model, alpha=0.5, label='EVP')

ax3.legend()
ax3.set_ylim(1e-5,8e-1)
ax3.set_xlim(0,0.14)
ax3.set_xlabel("time")
ax3.set_ylabel(r"$b_{rms}$")
mfig.figure.savefig("rbc_evp_ivp.png",dpi=300)




