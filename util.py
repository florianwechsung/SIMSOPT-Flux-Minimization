import numpy as np
from simsopt.geo.curve import curves_to_vtk
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.biotsavart import Current, Coil
from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.coilcollection import coils_via_symmetries
# from simsopt.geo.curveperturbed import CurvePerturbed, GaussianSampler
from simsopt._core.graph_optimizable import Optimizable

import coilpy

def get_NCSX_coils(coil_order, ppp, FIX_MODULAR_CURVE, FIX_MODULAR_CURRENT, FIX_CIRCULAR_CURVE, FIX_CIRCULAR_CURRENT):
    ################################################################################
    #### Read the ncsx coils #######################################################
    ################################################################################
    focuscoils = coilpy.coils.Coil().read_makegrid('coils.c09r00').data
    modular_curves = []
    modular_currents = []

    extra_curves = []
    extra_currents = []
    for i, c in enumerate(focuscoils):
        if i < 3:
            xyz = np.vstack((c.x, c.y, c.z)).T[:-1,:]
            n = xyz.shape[0]
            newcurve_ = CurveXYZFourier(np.linspace(0, 1, n, endpoint=False), coil_order)
            # newcurve_ = JaxCurveXYZFourier(np.linspace(0, 1, n, endpoint=False), coil_order)
            newcurve_.least_squares_fit(xyz)
            newcurve = CurveXYZFourier(np.linspace(0, 1, coil_order*ppp, endpoint=False), coil_order)
            newcurve.x = newcurve_.x
            current = Current(c.I)
            if FIX_MODULAR_CURRENT:
                current.fix_all()
            if FIX_MODULAR_CURVE:
                newcurve.fix_all()
            modular_curves.append(newcurve)
            modular_currents.append(current)
        elif i > 17:
            xyz = np.vstack((c.x, c.y, c.z)).T[:-1,:]
            n = xyz.shape[0]
            newcurve = CurveXYZFourier(np.linspace(0, 1, n, endpoint=False), 1)
            newcurve.least_squares_fit(xyz)
            newcurve.x = newcurve.get_dofs()
            current = Current(c.I)
            if FIX_CIRCULAR_CURVE:
                newcurve.fix_all()
            if FIX_CIRCULAR_CURRENT:
                current.fix_all()
            extra_curves.append(newcurve)
            extra_currents.append(current)


    modular_coils = coils_via_symmetries(modular_curves, modular_currents, 3, True)
    extra_coils = [Coil(curv, curr) for curv, curr in zip(extra_curves, extra_currents)]
    return modular_coils, extra_coils

def get_NCSX_plasma_field(nphi, ntheta):
    ################################################################################
    ##### Read in the B_plasma field on the boundary and the plasma shape ##########
    ################################################################################
    focusplasma = coilpy.read_focus_boundary('c09r00.boundary')
    nfp = 3
    phis = np.linspace(0, 1/(2*nfp), nphi)
    # phis = np.linspace(0, 1, 100)

    thetas = np.linspace(0, 1, ntheta)

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
    return s, Bplasma_n


################################################################################
#### Plot the B field ##########################################################
################################################################################
def plot_2d(Bcoil, unitn, Bplasma_n, filename=None):

    Bcoil_n = np.sum(Bcoil.B().reshape(unitn.shape)*unitn, axis=2)
    phis = np.linspace(0, 1/(2*3), Bplasma_n.shape[0])
    thetas = np.linspace(0, 1, Bplasma_n.shape[1])
    v_min = -0.15
    v_max = 0.15
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title(r'$B_\mathrm{P} \cdot n$')
    cs1 = ax1.contourf(2*np.pi*phis, 2*np.pi*thetas, Bplasma_n.T, vmin=v_min, vmax=v_max)
    # fig.colorbar(cs, ax=ax1)

    ax2.set_title(r'$B_\mathrm{BS} \cdot n$')
    cs2 = ax2.contourf(2*np.pi*phis, 2*np.pi*thetas, Bcoil_n.T, vmin=v_min, vmax=v_max)

    ax3.set_title(r'$B_\mathrm{P}\cdot n-B_\mathrm{BS}\cdot n $')
    cs3 = ax3.contourf(2*np.pi*phis, 2*np.pi*thetas, (Bplasma_n - Bcoil_n).T, vmin=v_min, vmax=v_max)

    plt.draw()
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(cs1, cax=cbar_ax)
    p1 = ax1.get_position().get_points().flatten()
    p3 = ax3.get_position().get_points().flatten()
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([p1[0], 0.05, p3[2]-p1[0], 0.10])
    fig.colorbar(cs1, cax=cbar_ax, orientation="horizontal")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
    plt.close()

def plot_3d(modular_curves, extra_curves, B_total, Bplasma_n, s, filename):
    n = s.normal()
    absn = np.linalg.norm(n, axis=2)[:, :, None]
    unitn = n/absn
    B = B_total.set_points(s.gamma().reshape((-1, 3))).B().reshape(s.gamma().shape)
    Bcoil_n = np.sum(B*unitn, axis=2)
    curves_to_vtk(modular_curves+extra_curves, filename)
    s.to_vtk(filename, {
        "B_BS": np.ascontiguousarray(Bcoil_n.reshape(absn.shape)),
        "B_P": np.ascontiguousarray(Bplasma_n.reshape(absn.shape)),
        "B_BS-B_P": np.ascontiguousarray((Bcoil_n-Bplasma_n).reshape(absn.shape))
    })

################################################################################
### Define an objetive that integrates 0.5 * |B\cdot n|^2 on a surface #########
################################################################################


class SquaredFlux(Optimizable):

    def __init__(self, surface, target, field):
        self.surface = surface
        self.target = target
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        Optimizable.__init__(self, x0=np.asarray([]), opts_in=[field])

    def J(self):
        xyz = self.surface.gamma()
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1./absn)[:,:,None]
        Bcoil = self.field.B().reshape(xyz.shape)
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        B_n = (Bcoil_n - self.target)
        return 0.5 * np.mean(B_n**2 * absn)

    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1./absn)[:,:,None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        B_n = (Bcoil_n - self.target)
        dJdB = (B_n[...,None] * unitn * absn[...,None])/absn.size
        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)
