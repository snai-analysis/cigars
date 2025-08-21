from typing import TYPE_CHECKING, Mapping

import torch
from pyro.distributions import Exponential, LogNormal, Normal, Uniform
from pyro.nn import PyroSample
from torch import Tensor

from clipppy.distributions import conundis
from clipppy.distributions.extra_dimensions import indep
from clipppy.utils.pyro import PyroDeterministic
from phytorch.cosmology.core import FLRW
from . import _PyroModule, SNDataT, GalaxyDataT, SPData


class HostConnection(_PyroModule):
    _priors = dict(
        gamma_M_logzsol=Uniform(-2, 2),
        # gamma_c_logzsol=Uniform(-0.05, 0.05),
        gamma_M_age=Uniform(-0.1, 0.1),

        step_M_logmass_loc=Uniform(9, 11),
        step_M_logmass=Uniform(-0.2, 0.2)
    )
    _inits = dict(
        # gamma_M_logzsol=0.3,  # MR+16
        gamma_M_logzsol=-0.1,  # Childress+13
        gamma_c_logzsol=0.02,  # Childress+13
        gamma_M_age=-0.03,

        step_M_logmass_loc=10.,
        step_M_logmass=-0.05
    )

    hostdata: GalaxyDataT
    spdata: SPData

    @PyroDeterministic
    def delta_M(self):
        return (
            self.gamma_M_logzsol * self.spdata['logzsol']
            + self.gamma_M_age * self.spdata['age']/1e9
            + torch.where(self.hostdata['logmass'] > self.step_M_logmass_loc, self.step_M_logmass, 0)
        )

    def forward(self):
        return {'delta_M': self.delta_M}

    if TYPE_CHECKING:
        def __call__(self) -> Mapping[str, Tensor]: ...


class SNPop(_PyroModule):
    N: int
    _priors = dict(
        M0=Uniform(-20, -19),
        sigma_res=Uniform(0.01, 0.2),
    )
    _inits = dict(
        M0=-19.5, sigma_res=0.1,
    )

    ext_keys = 'z',
    own_local_vars = 'z', 'delta_M', 'M_int'

    @PyroSample
    def z(self):
        return Uniform(0, 2).expand(torch.Size((self.N,))).to_event(1)

    @PyroSample
    def delta_M(self):
        return Uniform(-1, 1).expand(torch.Size((self.N,))).to_event(1)

    @property
    def _M_int(self):
        return self.M0.unsqueeze(-1) + self.delta_M

    @PyroSample
    def M_int(self):
        return Normal(self._M_int, self.sigma_res.unsqueeze(-1)).to_event(1)

    def forward(self):
        return {key: getattr(self, key) for key in self.ext_keys}

    if TYPE_CHECKING:
        def __call__(self) -> SNDataT: ...


class SimpleBayeSN(SNPop):
    _priors = SNPop._priors | dict(
        alpha=Uniform(-0.55, 0.25),
        alpha_c=Uniform(-0.03, 0.03),
        beta=Uniform(0, 4)
    )
    _inits = SNPop._inits | dict(
        alpha=-0.14, alpha_c=0, beta=2.2,
    )


    own_local_vars = SNPop.own_local_vars + ('x_int', 'c_int', 'R_V', 'A_V', 'E', 'M', 'x_ext', 'c_ext')

    mean_x = 0; sigma_x = 1
    mean_c = 0; sigma_c = 0.1

    tau = 0.1
    mean_RV = 3; sigma_RV = 0.5

    ext_keys = SNPop.ext_keys + ('M', 'x_ext', 'c_ext')

    @PyroSample
    def x_int(self):
        return indep(Normal(self.mean_x, self.sigma_x), (self.N,))

    @PyroSample
    def c_int(self):
        return Normal(self.mean_c + self.alpha_c.unsqueeze(-1) * self.x_int, self.sigma_c).to_event(1)

    @property
    def _M_int(self):
        return (
            super()._M_int
            + self.alpha.unsqueeze(-1) * self.x_int
            + self.beta.unsqueeze(-1) * self.c_int
        )

    @PyroSample
    def R_V(self):
        return indep(conundis.Normal(self.mean_RV, self.sigma_RV, constraint_lower=1.2), (self.N,))

    @PyroSample
    def A_V(self):
        return indep(Exponential(1/self.tau), (self.N,))

    @PyroDeterministic
    def E(self):
        return self.A_V / self.R_V

    @PyroDeterministic
    def M(self):
        return self.M_int + (self.R_V + 1) * self.E

    @PyroDeterministic
    def x_ext(self):
        return self.x_int

    @PyroDeterministic
    def c_ext(self):
        return self.c_int + self.E


class SNObs(_PyroModule):
    N: int
    cosmo: FLRW

    if TYPE_CHECKING:
        def __call__(self) -> SNDataT: ...


class ObservedSimpleBayeSN(SNObs):
    z: Tensor
    M: Tensor
    x_ext: Tensor
    c_ext: Tensor

    own_local_vars = 'm', 'Wmm', 'm_obs', 'Wxx', 'x_obs', 'Wcc', 'c_obs'

    @PyroDeterministic
    def mu(self):
        return self.cosmo.distmod(self.z)

    @PyroDeterministic
    def m(self):
        return self.M + self.mu

    @PyroSample
    def Wmm(self):
        return LogNormal(0.2*(self.m-56), 1.2).to_event(1)

    @PyroSample
    def m_obs(self):
        return Normal(self.m, self.Wmm**0.5).to_event(1)

    @PyroSample
    def Wxx(self):
        return LogNormal(-3, 1).expand_by(torch.Size((self.N,))).to_event(1)

    @PyroSample
    def x_obs(self):
        return Normal(self.x_ext, self.Wxx**0.5).to_event(1)

    @PyroSample
    def Wcc(self):
        return LogNormal(-7, 0.6).expand_by(torch.Size((self.N,))).to_event(1)

    @PyroSample
    def c_obs(self):
        return Normal(self.c_ext, self.Wcc**0.5).to_event(1)

    def forward(self):
        return {
            key: getattr(self, key)
            for key in ('m', 'x', 'c') for key in (f'{key}_obs', f'W{key}{key}')
        }
