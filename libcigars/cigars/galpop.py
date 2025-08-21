import attr
import torch

import phytorchx
from clipppy.utils.pyro import PyroDeterministic, Contextful
from phytorch.cosmology.core import FLRW
from phytorch.utils._typing import _t
from phytorchx.dataframe import TensorDataFrame
from slicsim.extinction import KriekConroy13
from . import _PyroModule, GalaxyDataT
from .. import lsst


@attr.s(eq=False, auto_attribs=True)
class GalaxyPopulation(_PyroModule):
    galpop: GalaxyDataT = attr.ib(factory=lambda: phytorchx.load('train/prospector-beta-sims.pt'))
    magerr: _t = 0.01

    cosmo: FLRW = attr.ib(init=False, repr=False)

    @staticmethod
    def calc_dust(data: GalaxyDataT):
        A_B, A_V = KriekConroy13().mag(data['dust_index'].new_tensor([4450, 5510])[:, None], data['dust_index'])
        return A_V/(A_B-A_V), data['dust2']

    @PyroDeterministic
    def gmags(self):
        mu = 5 * (self.cosmo.comoving_transverse_distance_dimless(self.galpop['zred']) * self.cosmo.hubble_distance_in_10pc).log10()
        return self.galpop['M'] + mu.unsqueeze(-1)

    @PyroDeterministic
    def gmags_obs(self):
        return self.gmags + self.magerr*torch.randn_like(self.gmags)

    own_local_vars = 'logmass', 'logzsol', 'gas_logz', 'dust2', 'dust_index', 'dust_ratio', 'gmags', 'gmags_obs'

    @Contextful
    def detgals(self):
        return TensorDataFrame(self.galpop.data | {
            'gmags': (gmags := self.gmags),
            'gmags_obs': self.gmags_obs
        })[
            (gmags < gmags.new_tensor(lsst.dethresh_coadd)).all(-1)
        ]
