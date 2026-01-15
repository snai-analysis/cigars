import attr
import pyro
import torch
from pyro.distributions import Normal
from pyro.nn import PyroSample

from clipppy.utils.pyro import Contextful
from phytorchx import mid_one
from .model import CIGARS
from .simple import ExtConstraintsMixin


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class CIGARSLikelihood(CIGARS, ExtConstraintsMixin):
    @Contextful
    def cosmoparams(self):
        return {
            key: pyro.sample(key, val)
            for key, val in self.cosmodel._priors.items()
        }

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
