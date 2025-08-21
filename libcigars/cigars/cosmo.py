from typing import Type, Iterable

import torch
from pyro.distributions import Uniform

from clipppy.utils.pyro import Contextful
from phytorch.cosmology.core import FLRW, H100
from phytorch.cosmology.drivers.analytic import LambdaCDM
from phytorch.cosmology.drivers.gridint import Flatw0waCDM
from phytorch.cosmology.module import AbstractCosmologyModule
from . import _PyroModule


class CosmologicalModel(_PyroModule):
    _cosmo_cls: Type[FLRW]
    _cosmo_params: Iterable[str]

    def __init__(self, h=0.7, **kwargs):
        super().__init__(**kwargs)
        self.cosmod = AbstractCosmologyModule(self._cosmo_cls, 1)
        self.cosmod.obj.H0 = h * H100

    @Contextful
    def cosmo_params(self):
        return {key: getattr(self, key) for key in self._cosmo_params}

    @Contextful
    def cosmo(self):
        return self.cosmod.set_params(**self.cosmo_params)


class LambdaCDMCosmo(CosmologicalModel):
    _cosmo_cls = LambdaCDM
    _priors = dict(
        Om0=Uniform(0, 1),
        Ode0=Uniform(0, 1),
    )
    _inits = dict(Om0=0.3, Ode0=0.7)
    _cosmo_params = _priors.keys()


class Flatw0waCDMCosmo(CosmologicalModel):
    _cosmo_cls = Flatw0waCDM
    _priors = dict(
        Om0=Uniform(0., 1.),
        w0=Uniform(-2., 0.),
        wa=Uniform(-3., 2.)
    )
    _inits = dict(Om0=0.3, w0=-1., wa=0.)
    _cosmo_params = _priors.keys()

    def __init__(self, h=0.7, **kwargs):
        super().__init__(h, **kwargs)
        self.cosmod.obj._set_grid(torch.linspace(0.001, 1.5, 1500))
