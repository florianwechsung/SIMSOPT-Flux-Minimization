from ncsx import CoilCollection
import numpy as np
try:
    from mayavi import mlab
    has_mayavi = True
except:
    has_mayavi = False

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.coilobjectives import CurveLength

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

stellarator = CoilCollection(modular_coils, modular_currents, 3, True)

################################################################################
##### Read in the B_plasma field on the boundary and the plasma shape ##########
################################################################################
focusplasma = coilpy.read_focus_boundary('c09r00.boundary')
nfp = 3
phis = np.linspace(0, 1/(2*nfp), 100)
# phis = np.linspace(0, 1, 100)

thetas = np.linspace(0, 1, 100)

# cheeky trick to get B on the surface: instead of reimplementing the fourier
# expansion, we make a SurfaceRZFourier, set the dofs for the z coordinate
# equal to the dofs of the Bfield and then just extract the z component of
# surface.gamma(). We never actually use this surface for anything, just as a
# means to turn fourier coeffiecients into actual Bfield values

mpol = max(focusplasma['bnormal']['xm'])
ntor = max(focusplasma['bnormal']['xn'])
# bnc = np.zeros((mpol+1, 2*ntor+1))
bns = np.zeros((mpol+1, 2*ntor+1))
for i, (m, n) in enumerate(zip(focusplasma['bnormal']['xm'], focusplasma['bnormal']['xn'])):
    # bnc[m, n+ntor] = focusplasma['bnormal']['bnc'][i]
    bns[m, n+ntor] = focusplasma['bnormal']['bns'][i]

s = SurfaceRZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=True, quadpoints_phi=phis, quadpoints_theta=thetas)
# s.zc[:,:] = bnc
s.zs[:,:] = bns
Bplasma_n = s.gamma()[:,:,2]

mpol = max(focusplasma['surface']['xm'])
ntor = max(focusplasma['surface']['xn'])
rc = np.zeros((mpol+1, 2*ntor+1))
zs = np.zeros((mpol+1, 2*ntor+1))
for i, (m, n) in enumerate(zip(focusplasma['surface']['xm'], focusplasma['surface']['xn'])):
    rc[m, n+ntor] = focusplasma['surface']['rbc'][i]
    zs[m, n+ntor] = focusplasma['surface']['zbs'][i]

s = SurfaceRZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=True, quadpoints_phi=phis, quadpoints_theta=thetas)
s.rc[:,:] = rc
s.zs[:,:] = zs


################################################################################
### Compute B field on the surface #############################################
################################################################################


xyz = s.gamma()
n = s.normal()
absn = np.linalg.norm(n, axis=2)
unitn = n * (1./absn)[:,:,None]

bs = BiotSavart(stellarator.coils, stellarator.currents)
bs.set_points(xyz.reshape((xyz.shape[0]*xyz.shape[1], 3)))
Bcoil_n = np.sum(bs.B().reshape(xyz.shape) * unitn, axis=2)

bsextra = BiotSavart(extra_coils, extra_currents)
B = bsextra.set_points(xyz.reshape((xyz.shape[0]*xyz.shape[1], 3))).B().reshape(xyz.shape)
Bextra_n = np.sum(B*unitn, axis=2)

################################################################################
#### Plot the B field ##########################################################
################################################################################
def plot_2d(Bcoil_n):
    v_min = -0.15
    v_max = 0.15
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title('Bplasma dot n')
    cs = ax1.contourf(2*np.pi*phis, 2*np.pi*thetas, Bplasma_n.T, vmin=v_min, vmax=v_max)
    fig.colorbar(cs, ax=ax1)

    ax2.set_title('Bbiotsavart dot n')
    cs = ax2.contourf(2*np.pi*phis, 2*np.pi*thetas, (Bcoil_n + Bextra_n).T, vmin=v_min, vmax=v_max)

    ax3.set_title('Combined')
    cs = ax3.contourf(2*np.pi*phis, 2*np.pi*thetas, (Bplasma_n - Bcoil_n - Bextra_n).T, vmin=v_min, vmax=v_max)
    plt.show()

def plot_3d(B):
    if not has_mayavi:
        return
    for c in stellarator.coils:
        c.plot_mayavi(show=False)
    for c in extra_coils:
        c.plot_mayavi(show=False)
    s.plot(show=False, scalars=B, wireframe=False)
    mlab.colorbar()
    mlab.show()

plot_2d(Bcoil_n)
plot_3d(Bplasma_n - (Bcoil_n + Bextra_n))

################################################################################
### Define an objetive that integrates 0.5 * |B\cdot n|^2 on a surface #########
################################################################################

coil_lengths = [CurveLength(c) for c in modular_coils]

alpha = 1e-4

def fun(dofs):
    stellarator.set_dofs(dofs)
    Bcoil = bs.set_points(xyz.reshape((xyz.shape[0]*xyz.shape[1], 3))).B().reshape(xyz.shape)
    Bcoil_n = np.sum(Bcoil*unitn, axis=2)
    B_n = (Bcoil_n + Bextra_n) - Bplasma_n
    J = 0.5 * np.mean(B_n**2 * absn)
    dJdB = (B_n[...,None] * unitn * absn[...,None])/absn.size
    dJdB = dJdB.reshape((dJdB.shape[0]*dJdB.shape[1], 3))
    empty = np.zeros((dJdB.shape[0], 3, 3))
    deriv = bs.B_vjp(dJdB)
    grad = stellarator.reduce_coefficient_derivatives(deriv)
    print(f"J={J}")
    if alpha > 0:
        J += alpha * sum([J.J() for J in coil_lengths])
        grad += alpha * np.concatenate([J.dJ() for J in coil_lengths])
    return J, grad
print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
dofs = stellarator.get_dofs()
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = fun(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    J1, _ = fun(dofs + eps*h)
    print("err", (J1-J0)/eps - dJh)
print("""    
################################################################################
### Run some optimisation ######################################################
################################################################################
""")
from scipy.optimize import minimize
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 50, 'maxcor': 400}, tol=1e-15)
Bcoil_n = np.sum(bs.set_points(xyz.reshape((xyz.shape[0]*xyz.shape[1], 3))).B().reshape(xyz.shape) * unitn, axis=2)
plot_2d(Bcoil_n)
plot_3d(Bplasma_n - (Bcoil_n + Bextra_n))
