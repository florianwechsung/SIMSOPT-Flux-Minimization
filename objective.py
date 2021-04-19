from simsopt.core.optimizable import Optimizable
from simsopt.geo.biotsavart import BiotSavart
import numpy as np

class QuadraticFlux(Optimizable):

    def __init__(self, stellarator, Btarget_n, surface):
        self.stellarator = stellarator
        self.bs = BiotSavart(stellarator.coils, stellarator.currents)
        self.Btarget_n = Btarget_n
        # self.depends_on = [stellarator]
        self.depends_on = stellarator.coils
        self.surface = surface

    def J(self):
        xyz = self.surface.gamma()
        Bcoil = self.bs.set_points(xyz.reshape((-1, 3))).B().reshape(xyz.shape)
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n/absn[..., None]
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        B_n = Bcoil_n - self.Btarget_n
        J = 0.5 * np.mean(B_n**2 * absn)
        return J

    def dJ(self):
        xyz = self.surface.gamma()
        Bcoil = self.bs.set_points(xyz.reshape((-1, 3))).B().reshape(xyz.shape)
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n/absn[..., None]
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        B_n = Bcoil_n - self.Btarget_n


        # dJ_dcoil
        dJdB = (B_n[..., None] * unitn * absn[..., None])/absn.size
        dJdB = dJdB.reshape((-1, 3))
        deriv = self.bs.B_vjp(dJdB)
        grad = self.stellarator.reduce_coefficient_derivatives(deriv)
        return grad

        # # dJ_dcurrent
        # dB_by_dcoilcurrents = self.bs.dB_by_dcoilcurrents()
        # ncoils = len(dB_by_dcoilcurrents)
        # grad_current = np.zeros((ncoils, ))
        # for i in range(ncoils):
        #     dBdI = dB_by_dcoilcurrents[i].reshape(xyz.shape)
        #     grad_current[i] = np.sum((dBdI*unitn)*B_n[..., None]*absn[..., None])/absn.size
        # grad_current = self.stellarator.reduce_current_derivatives(grad_current)

        # return np.concatenate((grad, grad_current))
