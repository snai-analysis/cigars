from abc import abstractmethod

import torch
from math import log
from pyro.distributions import Normal
from pyro.nn import pyro_method
from torch import LongTensor

from clipppy.utils.pyro import PyroDeterministic
from phytorchx.dataframe import TensorDataFrame
from phytorch.utils._typing import _t
from . import _PyroModule, GalaxyDataT


class SNOcc(_PyroModule):
    hostpop: GalaxyDataT

    @abstractmethod
    def sample(self, N: int) -> tuple[GalaxyDataT, tuple[LongTensor, LongTensor]]: ...

    @property
    @abstractmethod
    def expected_count(self) -> _t: ...


class DTD(SNOcc):
    _priors = dict(
        # dtd_A=conundis.Normal(-12.15, 0.1, constraint_lower=-12.5, constraint_upper=-11.8),
        # dtd_s=conundis.Normal(-1.34, 0.2, constraint_lower=-2., constraint_upper=-0.7),
        dtd_A=Normal(-12.15, 0.1),
        dtd_s=Normal(-1.34, 0.2)
    )
    _inits = dict(dtd_A=-12.15, dtd_s=-1.34)

    @PyroDeterministic
    def lnrate(self):
        ret = log(10) * (
            self.dtd_A[..., None, None] - (1+self.hostpop['zred']).log10().unsqueeze(-1)
            + (self.hostpop['logmasses'][..., 2:] + self.dtd_s[..., None, None] * (self.hostpop['logages'][..., 2:] - 9))
        )
        if '_lnweight' in self.hostpop:
            ret = ret + self.hostpop['_lnweight'].unsqueeze(-1)
        return ret

    @PyroDeterministic
    def expected_count(self):
        return self.lnrate.flatten(-2).logsumexp(-1).exp()

    def stats(self):
        lnNg = (lnN := self.lnrate).logsumexp(-1)
        return (
            lnN.flatten(-2).logsumexp(-1).exp().item(),
            (2*lnNg.logsumexp(0) - (2*lnNg).logsumexp(0)).exp().item()
        )

    @pyro_method
    def sample(self, N: int):
        p = torch.distributions.Categorical(logits=self.lnrate.flatten(-2))
        host_idx, sp_idx = torch.functional.unravel_index(p.sample(torch.Size((N,))), self.lnrate.shape[-2:])

        return TensorDataFrame(self.hostpop[host_idx]), (host_idx, sp_idx+2)
