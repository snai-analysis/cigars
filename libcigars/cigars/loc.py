from abc import abstractmethod
from pathlib import Path

import attr
from pyro.distributions import Categorical, MixtureSameFamily
from pyro.nn import pyro_method, PyroSample
from torch import Tensor

import phytorchx
from clipppy.distributions import conundis
from clipppy.distributions.concrete import GeneralizedGammaDistribution
from clipppy.utils.pyro import PyroDeterministic
from phytorch.interpolate import Linear1dInterpolator
from phytorch.special.gammainc import gammaincinv
from . import _PyroModule, GalaxyDataT


class SersicDistribution(GeneralizedGammaDistribution):
    @staticmethod
    def bn(n, d=2):
        return gammaincinv(d*n, 0.5)

    def __init__(self, n, re=1., d=2):
        self.n, self.re = n, re
        super().__init__(self.re / self.bn(self.n, d)**self.n, 1/self.n, d)

    def log_rel_density(self, r):
        return self.log_prob(r) - (self.d-1) * r.log()


class SNLoc(_PyroModule):
    hostdata: GalaxyDataT
    R_Vg: Tensor
    A_Vg: Tensor

    @abstractmethod
    def dust(self) -> tuple[Tensor, Tensor]: ...


@attr.s(eq=False, auto_attribs=True)
class DustPediaSNLoc(SNLoc):
    stars_sersic_re = 1.
    dust_sersic_re = 1.

    mixture_params: Linear1dInterpolator = attr.ib(factory=lambda: phytorchx.load(Path(__file__).parent / 'dustpedia.pt')['sersic_ns'])

    @PyroSample
    def stars_sersic_n(self):
        weights, means, stds = self.mixture_params(
            self.hostdata['logmass'].clip(*self.mixture_params.x[(0, -1),])
        ).unbind(-2)
        return MixtureSameFamily(Categorical(probs=weights), conundis.Normal(means, stds, constraint_lower=0.5)).to_event(1)

    @PyroSample
    def loc(self):
        return SersicDistribution(self.stars_sersic_n, self.stars_sersic_re).to_event(1)

    @staticmethod
    def _C_dustnorm(ns):
        return 0.0096 * ns**3 - 0.021 * ns**2 + 0.080 * ns + 0.64

    @PyroDeterministic
    def dust_sersic_n(self):
        return self.stars_sersic_n / 2

    @pyro_method
    def dust_profile(self, r: Tensor):
        return SersicDistribution(self.dust_sersic_n, self.dust_sersic_re).log_rel_density(r).exp()

    @pyro_method
    def dust(self):
        return (
            self.R_Vg,
            self.A_Vg / self._C_dustnorm(self.stars_sersic_n) * self.dust_profile(self.loc)
        )
