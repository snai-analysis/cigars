from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

from torch import Tensor

import phytorchx
from phytorch.cosmology.core import FLRW
from slicsim.bandpasses import lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y
from slicsim.bandpasses.bandpass import Bandpass
from slicsim.bandpasses.magsys import MagSys, AB
from slicsim.effects import affected, Distance, CosmologicalDistance, Redshifted
from slicsim.model import LightcurveModel, Field


@dataclass
class Galaxy:
    bands: Sequence[Bandpass] = lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y
    magsys: MagSys = AB
    spa_subs: float = 2

    def __post_init__(self):
        self.field = Field(times=[0]*len(self.bands), bands=self.bands, magsys=self.magsys)

    @cached_property
    def src(self):
        from .source import SpeculatorAlphaSource
        from .torchspec import SpeculatorAlpha

        return SpeculatorAlphaSource(SpeculatorAlpha(phytorchx.load(
            Path(__file__).parent / 'speculator_alpha.pt'
        ).subs[:self.spa_subs]))

    def model(self, params: Mapping[str, Tensor], cosmo: FLRW = None):
        return LightcurveModel(affected(
            self.src,
            Distance() if cosmo is None else
            CosmologicalDistance(cosmo=cosmo, z_cosmo=params['zred'].unsqueeze(-1)),
            Redshifted(z=params['zred'].unsqueeze(-1)),
        ), self.field)

    def generate(self, params: Mapping[str, Tensor], cosmo: FLRW = None):
        return -2.5 * self.model(params, cosmo).bandcountscal(**params).log10()
