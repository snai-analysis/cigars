from collections import ChainMap
from contextlib import nullcontext
from enum import Enum, auto
from math import log, exp
from typing import Collection, Iterable, Mapping, Literal

import attr
import pyro
import torch
from pyro.distributions import Bernoulli, Poisson
from pyro.nn import PyroSample
from torch import Tensor, BoolTensor

from phytorchx.dataframe import TensorDataFrame
from phytorchx.vltensor import VLTensor
from . import _PyroModule, SPData, GalaxyDataT, SNDataT
from .cosmo import CosmologicalModel, LambdaCDMCosmo
from .detprob import SNDetProb, ToyLSSTSNDetProb
from .galpop import GalaxyPopulation
from .loc import SNLoc, DustPediaSNLoc
from .snocc import SNOcc, DTD
from .snpop import SNPop, SimpleBayeSN, SNObs, ObservedSimpleBayeSN, HostConnection


@attr.s(eq=False, auto_attribs=True)
class CIGARS(_PyroModule):
    counts: int = 1000

    cosmodel: CosmologicalModel = attr.ib(factory=LambdaCDMCosmo)

    galpop: GalaxyPopulation = attr.ib(factory=GalaxyPopulation)

    class DustModel(Enum):
        independent = auto()
        match_global = auto()
        match_local = auto()

    dust_model: DustModel = DustModel.independent

    snocc: SNOcc = attr.ib(factory=DTD)
    snloc: SNLoc = attr.ib(factory=DustPediaSNLoc)
    snhostconn: HostConnection = attr.ib(factory=HostConnection)
    snpop: SNPop = attr.ib(factory=SimpleBayeSN)
    snobs: SNObs = attr.ib(factory=ObservedSimpleBayeSN)

    sndetprob: SNDetProb = attr.ib(factory=ToyLSSTSNDetProb)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.counts_scale = (self.counts / 500) / (len(self.galpop.galpop) / 1e6)

        if '_lnweight' in self.galpop.galpop:
            self.counts_scale /= exp(
                self.galpop.galpop['_lnweight'].logsumexp(-1).item()
                - log(len(self.galpop.galpop['_lnweight']))
            )

    @PyroSample
    def sncount(self):
        return Poisson(self.counts_scale * self.snocc.expected_count)

    def forward(self, endat: Literal['host'] = None):
        cosmo = self.cosmodel.cosmo

        self.snocc.set(hostpop=TensorDataFrame(self.galpop.set(cosmo=cosmo).detgals))
        N = int(self.sncount)
        hostdata, (host_idx, sp_idx) = self.snocc.sample(N)

        if endat == 'host':
            return self.snocc.hostpop, hostdata, (host_idx, sp_idx)

        conds = {'z': hostdata['zred']}

        spdata = SPData(
            age=torch.lerp(
                10**hostdata['agebins'].take_along_dim(sp_idx.unsqueeze(-1), -1).squeeze(-1),
                10**hostdata['agebins'].take_along_dim(1+sp_idx.unsqueeze(-1), -1).squeeze(-1),
                torch.rand_like(sp_idx, dtype=hostdata['agebins'].dtype)
            ),
            logzsol=hostdata['logzsol']
        )
        conds.update(self.snhostconn.set(hostdata=hostdata, spdata=spdata)())

        if self.dust_model is self.DustModel.independent:
            pass
        else:
            R_Vg, A_Vg = self.galpop.calc_dust(hostdata)

            if self.dust_model is self.DustModel.match_global:
                conds['R_V'], conds['A_V'] = R_Vg, A_Vg

            elif self.dust_model is self.DustModel.match_local:
                self.snloc.set(hostdata=hostdata, R_Vg=R_Vg, A_Vg=A_Vg)
                conds['R_V'], conds['A_V'] = self.snloc.dust()

        with pyro.condition(data={
            self.snpop._pyro_get_fullname(key): val
            for key, val in conds.items()
        }):
            snpop = self.snpop.set(N=N)()

        sndata = self.snobs.set(cosmo=cosmo, N=N, **snpop)()

        return hostdata, TensorDataFrame(spdata | snpop | sndata | {
            'snsel': pyro.sample(
                'snsel',
                Bernoulli(self.sndetprob.detprob(hostdata, sndata)).to_event(1)
            ).to(bool)
        })

    def _generate(self, initting=False, seed=None) -> tuple[Mapping[str, Tensor], GalaxyDataT, SNDataT, BoolTensor, Tensor]:
        if seed is not None:
            torch.manual_seed(seed)

        with self.init if initting else nullcontext():
            trace = pyro.poutine.trace(self).get_trace()

        trace_data = {key.rsplit('.', 1)[-1]: val['value'] for key, val in trace.nodes.items() if not key.startswith('_')}
        hostdata, sndata = trace.nodes['_RETURN']['value']
        sel = sndata['snsel'].to(bool)
        selprob = trace.nodes['snsel']['fn'].base_dist.probs

        return trace_data, hostdata, sndata, sel, selprob

    @staticmethod
    def _generate_postprocess(res, keys_global, keys_local):
        trace_data, hostdata, sndata, sel, selprob = res
        popdata = ChainMap(sndata, hostdata, trace_data)
        return {key: trace_data[key] for key in keys_global} | {
            key: popdata[key][sel].as_subclass(VLTensor)
            for key in keys_local
        }

    def generate(self, keys_global: Iterable[str], keys_local: Collection[str], initting=False):
        return self._generate_postprocess(self._generate(initting=initting), keys_global, keys_local)
