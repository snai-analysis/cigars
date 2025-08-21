from math import log, pi

from torch import Tensor

from phytorch.constants import c as speed_of_light
from phytorch.interpolate import Linear1dInterpolator
from phytorch.units.astro import jansky, pc
from phytorch.units.si import angstrom
from slicsim.sources.abc import ConstantSource
from slicsim.utils import _t

from .torchspec import SpeculatorAlpha


class SpeculatorAlphaSource(ConstantSource):
    def __init__(self, spa: SpeculatorAlpha):
        self.spa = spa
        self._log_flux_offset = 2 * self.spa.wavelengths.log()

    logmass: Tensor
    flux_unit = (3631*jansky * 4*pi*(10*pc)**2) * speed_of_light / angstrom**2

    def flux(self, wave: _t, **kwargs) -> _t:
        return Linear1dInterpolator(
            self.spa.wavelengths,
            (self.spa(kwargs) + log(10)*self.logmass.unsqueeze(-1) - self._log_flux_offset).exp().unsqueeze(-2),
            channel_ndim=0
        )(wave)
