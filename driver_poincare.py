from ncsx import CoilCollection
from poincareplot import compute_field_lines
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.objectives import CurveLength
import matplotlib.pyplot as plt

import coilpy

################################################################################
#### Read the ncsx coils #######################################################
################################################################################
focuscoils = coilpy.coils.Coil().read_makegrid('coils.c09r00').data
modular_coils = []
modular_currents = []
coil_order = 20

extra_coils = []
extra_currents = []
for i, c in enumerate(focuscoils):
    if i < 3:
        xyz = np.vstack((c.x, c.y, c.z)).T[:-1,:]
        n = xyz.shape[0]
        newcoil = CurveXYZFourier(np.linspace(0, 1, n, endpoint=False), coil_order)
        newcoil.least_squares_fit(xyz)
        modular_coils.append(newcoil)
        modular_currents.append(c.I)
    elif i > 17:
        xyz = np.vstack((c.x, c.y, c.z)).T[:-1,:]
        n = xyz.shape[0]
        newcoil = CurveXYZFourier(np.linspace(0, 1, n, endpoint=False), 1)
        newcoil.least_squares_fit(xyz)
        extra_coils.append(newcoil)
        extra_currents.append(c.I)

nfp = 3
stellarator = CoilCollection(modular_coils, modular_currents, nfp, True)
biotsavart = BiotSavart(stellarator.coils + extra_coils, stellarator.currents + extra_currents)

################################################################################
################################################################################
################################################################################
outdir = 'output/'
import os
os.makedirs(outdir, exist_ok=True)

nperiods = 200
magnetic_axis_radius=1.55

spp = 120
rphiz, xyz, absB, phi_no_mod = compute_field_lines(biotsavart, nperiods=nperiods, batch_size=4, magnetic_axis_radius=magnetic_axis_radius, max_thickness=0.2, delta=0.01, steps_per_period=spp)
nparticles = rphiz.shape[0]

data0 = np.zeros((nperiods, nparticles*2))
data1 = np.zeros((nperiods, nparticles*2))
data2 = np.zeros((nperiods, nparticles*2))
data3 = np.zeros((nperiods, nparticles*2))
for i in range(nparticles):
    data0[:, 2*i+0] = rphiz[i, range(0, nperiods*spp, spp), 0]
    data0[:, 2*i+1] = rphiz[i, range(0, nperiods*spp, spp), 2]
    data1[:, 2*i+0] = rphiz[i, range(1*spp//(nfp*4), nperiods*spp, spp), 0]
    data1[:, 2*i+1] = rphiz[i, range(1*spp//(nfp*4), nperiods*spp, spp), 2]
    data2[:, 2*i+0] = rphiz[i, range(2*spp//(nfp*4), nperiods*spp, spp), 0]
    data2[:, 2*i+1] = rphiz[i, range(2*spp//(nfp*4), nperiods*spp, spp), 2]
    data3[:, 2*i+0] = rphiz[i, range(3*spp//(nfp*4), nperiods*spp, spp), 0]
    data3[:, 2*i+1] = rphiz[i, range(3*spp//(nfp*4), nperiods*spp, spp), 2]

np.savetxt(outdir + 'poincare0.txt', data0, comments='', delimiter=',')
np.savetxt(outdir + 'poincare1.txt', data1, comments='', delimiter=',')
np.savetxt(outdir + 'poincare2.txt', data2, comments='', delimiter=',')
np.savetxt(outdir + 'poincare3.txt', data3, comments='', delimiter=',')
modBdata = np.hstack((phi_no_mod[:, None], absB.T))[0:(10*spp)]
np.savetxt(outdir + 'modB.txt', modBdata, comments='', delimiter=',')
plt.figure()
for i in range(min(modBdata.shape[1]-1, 10)):
    plt.plot(modBdata[:, 0], modBdata[:, i+1], zorder=100-i)
plt.savefig(outdir + "absB.png", dpi=300)
plt.close()
import mayavi.mlab as mlab
mlab.options.offscreen = True
for coil in stellarator.coils + extra_coils:
    mlab.plot3d(coil.gamma()[:, 0], coil.gamma()[:, 1], coil.gamma()[:, 2], color=(0., 0., 0.))
colors = [
    (0.298, 0.447, 0.690), (0.866, 0.517, 0.321), (0.333, 0.658, 0.407), (0.768, 0.305, 0.321),
    (0.505, 0.447, 0.701), (0.576, 0.470, 0.376), (0.854, 0.545, 0.764), (0.549, 0.549, 0.549),
    (0.800, 0.725, 0.454), (0.392, 0.709, 0.803)
]
counter = 0
for i in range(0, nparticles):
    mlab.plot3d(xyz[i, :, 0], xyz[i, :, 1], xyz[i, :, 2], tube_radius=0.005, color=colors[counter%len(colors)])
    counter += 1
mlab.view(azimuth=0, elevation=0)
mlab.savefig(outdir + "poincare-3d.png", magnification=4)
mlab.close()

for k in range(4):
    plt.figure()
    for i in range(nparticles):
        plt.scatter(rphiz[i, range(k * spp//(nfp*4), nperiods*spp, spp), 0], rphiz[i, range(k * spp//(nfp*4), nperiods*spp, spp), 2], s=0.1)
    plt.savefig(outdir + f"poincare_{k}.png", dpi=300)
    plt.close()
