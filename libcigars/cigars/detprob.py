from abc import ABC
from pathlib import Path

import torch
from torch import Tensor

import phytorchx
from phytorch.interpolate import LinearNDGridInterpolator
from . import GalaxyDataT, SNDataT


class SNDetProb(ABC):
    def detprob(self, hostdata: GalaxyDataT, sndata: SNDataT) -> Tensor: ...


class ToyLSSTSNDetProb(SNDetProb):
    _detprob = LinearNDGridInterpolator(*phytorchx.load(Path(__file__).parent / 'detprob.pt'))

    def detprob(self, hostdata: GalaxyDataT, sndata: SNDataT) -> Tensor:
        return self._detprob(torch.stack((hostdata['zred'], sndata['m_obs']), -1))
