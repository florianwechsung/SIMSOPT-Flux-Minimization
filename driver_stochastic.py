from util import get_NCSX_coils, get_NCSX_plasma_field, plot_2d, plot_3d, SquaredFlux
import numpy as np

from simsopt.geo.biotsavart import BiotSavart, Current, Coil
from simsopt.geo.curveobjectives import CurveLength
from simsopt._core.graph_optimizable import Optimizable
from simsopt._core.derivative import Derivative
from curveperturbed import CurvePerturbed, GaussianSampler


# Pick which parts of NCSX we want to optimize for. e.g. we could keep the coil shape constant, but only optimize for currents
# or keep the circular coils constant, and optimize for shape and current of the modular coils
FIX_MODULAR_CURRENT = False # keep currents of modular coils fixed
FIX_MODULAR_CURVE = False # keep shape of modular coils fixed
FIX_CIRCULAR_CURRENT = True # keep currents in circular coils fixed
FIX_CIRCULAR_CURVE = True # keep shape of circular coils fixed

ALPHA = 1e-6 # Total objective is SquaredFlux + alpha * sum(coil_lengths)
COIL_ORDER = 15 # max order of modular coils
PPP = 20 # points-per-period quadrature order for modular coils
NPHI = 64 # number of quadrature points in phi on surface
NTHETA = 64 # number of quadrature points in theta on surface

modular_coils, extra_coils = get_NCSX_coils(COIL_ORDER, PPP, FIX_MODULAR_CURVE, FIX_MODULAR_CURRENT, FIX_CIRCULAR_CURVE, FIX_CIRCULAR_CURRENT)
s, Bplasma_n = get_NCSX_plasma_field(NPHI, NTHETA)


################################################################################
### Compute B field on the surface #############################################
################################################################################

xyz = s.gamma()
n = s.normal()
absn = np.linalg.norm(n, axis=2)[:, :, None]
unitn = n/absn

bs = BiotSavart(modular_coils)
bs_extra = BiotSavart(extra_coils)

B_total = bs + bs_extra
B_total.set_points(xyz.reshape((-1, 3)))
B_total_n = np.sum(B_total.B().reshape(xyz.shape) * unitn, axis=2)

modular_curves = [c.curve for c in modular_coils]
extra_curves = [c.curve for c in extra_coils]

plot_2d(B_total, unitn, Bplasma_n, filename='/tmp/init.png')
plot_3d(modular_curves, extra_curves, B_total, Bplasma_n, s, '/tmp/init')


class StochasticFOCUSObjective(Optimizable):

    def __init__(self, Jfluxs, Jcls, alpha):
        Optimizable.__init__(self, x0=np.asarray([]), opts_in=Jfluxs + Jcls)
        self.Jfluxs = Jfluxs
        self.Jcls = Jcls
        self.alpha = alpha

    def J(self):
        self.vals = [np.mean([Jflux.J() for Jflux in self.Jfluxs])] + [Jcl.J() for Jcl in self.Jcls]
        return self.vals[0] + self.alpha * sum(self.vals[1:])

    def dJ(self):
        res = Derivative({})
        for JF in self.Jfluxs:
            res += (1./len(self.Jfluxs)) * JF.dJ()
        for Jcl in self.Jcls:
            res += self.alpha * Jcl.dJ()
        return res


Jflux = SquaredFlux(s, Bplasma_n, B_total)
coil_lengths = [CurveLength(c) for c in modular_curves]

sampler = GaussianSampler(modular_curves[0].quadpoints, 0.003, 0.3, n_derivs=1)
Nsamples = 16
Jflux_pert = []
for i in range(2*Nsamples):
    perturbed_modular_coils = [Coil(CurvePerturbed(mc.curve, sampler), mc.current) for mc in modular_coils]
    B_total_pert = BiotSavart(perturbed_modular_coils) + bs_extra
    Jflux_pert.append(SquaredFlux(s, Bplasma_n, B_total_pert))


JF = StochasticFOCUSObjective(Jflux_pert[:Nsamples], coil_lengths, ALPHA)
JF_oos = StochasticFOCUSObjective(Jflux_pert[Nsamples:], coil_lengths, ALPHA)

def fun(dofs):
    JF.x = dofs
    JF_oos.x = dofs
    J = JF.J()
    J_oos = JF_oos.J()
    dJ = JF.dJ()
    grad = dJ(JF)
    print(f"J={J:.3e}, J_oos={J_oos:.3e}, Jflux={JF.vals[0]:.3e}, CoilLengths=[{JF.vals[1]:.3f}, {JF.vals[2]:.3f}, {JF.vals[3]:.3f}], ||∇J||={np.linalg.norm(grad):.3e}")
    return J, grad

dofs = JF.x
print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = fun(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    J1, _ = fun(dofs + eps*h)
    J2, _ = fun(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)
print("""    
################################################################################
### Run some optimisation ######################################################
################################################################################
""")
from scipy.optimize import minimize
print("fun")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 100, 'maxcor': 400}, tol=1e-15)

plot_2d(B_total, unitn, Bplasma_n, filename='/tmp/stoch.png')
plot_3d(modular_curves, extra_curves, B_total, Bplasma_n, s, '/tmp/stoch')
