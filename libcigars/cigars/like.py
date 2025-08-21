import attr
import pyro
import torch
from pyro.distributions import Normal
from pyro.nn import pyro_method, PyroSample

from clipppy.utils.pyro import Contextful
from phytorch.constants import c as speed_of_light
from phytorch.cosmology.special.decdm import BaseDECDM
from phytorch.units import Unit
from phytorchx import mid_one
from .model import CIGARS


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class CIGARSLikelihood(CIGARS):
    cosmo_planck: BaseDECDM
    cosmo_lowz: BaseDECDM

    @Contextful
    def cosmoparams(self):
        return {
            key: pyro.sample(key, val)
            for key, val in self.cosmodel._priors.items()
        }

    planck_z: float = 1100.
    planck_sigma: float = 0.006

    @PyroSample
    def planck(self):
        self.cosmo_planck._set_params(**self.cosmoparams)
        return Normal(
            (self.cosmo_planck.Om0**0.5 * self.cosmo_planck.H0 / speed_of_light
             * (1+self.planck_z) * self.cosmo_planck.angular_diameter_distance(self.planck_z)
             ).to(Unit()).value,
            self.planck_sigma
        )

    lowz_z: float = 0.05
    lowz_sigma: int = 0.1 / 500**0.5

    @PyroSample
    def lowz(self):
        return Normal(self.cosmo_planck._set_params(**self.cosmoparams).distmod(self.lowz_z), self.lowz_sigma)

    highz_cond: dict

    def __attrs_post_init__(self):
        self.mgrid = mid_one(torch.linspace(12, 26, 1401))
        self.lnprob_sel = self.sndetprob._detprob(
            torch.stack(torch.broadcast_tensors(self.highz_cond['z'][:, None], self.mgrid[None, :]), -1)
        ).log()

    @PyroSample
    def highz(self):
        cosmo = self.cosmodel.cosmod.set_params(**self.cosmoparams)

        M0 = pyro.sample('M0', self.snpop._priors['M0']).unsqueeze(-1)
        sigma_res = pyro.sample('sigma_res', self.snpop._priors['sigma_res']).unsqueeze(-1)

        alpha = pyro.sample('alpha', self.snpop._priors['alpha']).unsqueeze(-1)
        beta = pyro.sample('beta', self.snpop._priors['beta']).unsqueeze(-1)

        m_mean = (
            M0 + cosmo.distmod(self.highz_cond['z'])
            + self.highz_cond['delta_M']
            + alpha * self.highz_cond['x_int'] + beta * self.highz_cond['c_int'] + ((self.highz_cond['R_V'] + 1) * self.highz_cond['A_V'] / self.highz_cond['R_V'])
        )

        like = Normal(m_mean, (self.highz_cond['Wmm'] + sigma_res ** 2) ** 0.5)

        lnprob_grid = like.log_prob(self.mgrid[..., None, None]).movedim(0, -1)
        pyro.factor('selcorr', - ((lnprob_grid + self.lnprob_sel).logsumexp(-1) - lnprob_grid.logsumexp(-1)).sum(-1))

        return like.to_event(1)
