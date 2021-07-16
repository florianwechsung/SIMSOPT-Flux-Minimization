import numpy as np
import simsoptpp as sopp
from simsopt.geo.curve import Curve
from sympy import Symbol, lambdify, exp
import jax.numpy as jnp


class GaussianSampler():

    def __init__(self, points, sigma, length_scale, n_derivs=1):
        self.points = points
        xs = self.points
        n = len(xs)
        self.n_derivs = n_derivs
        cov_mat = np.zeros((n*(n_derivs+1), n*(n_derivs+1)))

        def kernel(x, y):
            return sum((sigma**2)*exp(-(x-y+i)**2/(length_scale**2)) for i in range(-2, 3))
        XX, YY = np.meshgrid(xs, xs)
        for i in range(n):
            for j in range(n):
                x = XX[i, j]
                y = YY[i, j]
                if abs(x-y) > 0.5:
                    if y > 0.5:
                        YY[i, j] -= 1
                    else:
                        XX[i, j] -= 1

        for ii in range(n_derivs+1):
            for jj in range(n_derivs+1):
                x = Symbol("x")
                y = Symbol("y")
                f = kernel(x, y)
                if ii + jj == 0:
                    lam = lambdify((x, y), f, "numpy")
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX, YY)

        from scipy.linalg import sqrtm
        self.L = np.real(sqrtm(cov_mat))

    def sample(self, randomgen=None):
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]


class CurvePerturbed(sopp.Curve, Curve):

    def __init__(self, curve, sampler, randomgen=None):
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, x0=np.asarray([]), opts_in=[curve])
        self.sampler = sampler
        self.randomgen = randomgen
        self.sample = sampler.sample(self.randomgen)

    def resample(self):
        self.sample = self.sampler.sample(self.randomgen)
        self.invalidate_cache()

    def get_dofs(self):
        return np.asarray([])

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.curve.gamma() + self.sample[0]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        dgamma_by_dcoeff[:] = self.curve.dgamma_by_dcoeff()

    def dgammadash_by_dcoeff_impl(self, dgammadash_by_dcoeff):
        dgammadash_by_dcoeff[:] = self.curve.dgammadash_by_dcoeff()

    def dgammadashdash_by_dcoeff_impl(self, dgammadashdash_by_dcoeff):
        dgammadashdash_by_dcoeff[:] = self.curve.dgammadashdash_by_dcoeff()

    def dgammadashdashdash_by_dcoeff_impl(self, dgammadashdashdash_by_dcoeff):
        dgammadashdashdash_by_dcoeff[:] = self.curve.dgammadashdashdash_by_dcoeff()

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgamma_by_dcoeff_vjp_graph(self, v):
        return self.curve.dgamma_by_dcoeff_vjp_graph(v)

    def dgammadash_by_dcoeff_vjp_graph(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp_graph(v)
