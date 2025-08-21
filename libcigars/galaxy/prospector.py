from functools import partial
from math import log, log10
from typing import Iterable

import numpy as np
import torch
from prospect.models import priors_beta
from pyro.distributions import Uniform, StudentT, TransformedDistribution
from pyro.nn import PyroSample, PyroModule, pyro_method
from torch import Tensor
from torch.distributions import ExpTransform
from tqdm.auto import tqdm

import phytorchx
from clipppy.distributions import conundis
from clipppy.distributions.histdist import HistDist
from clipppy.utils.pyro import PyroDeterministic
from phytorch.math import exp


def schechter(logm, logphi, logmstar, alpha):
    return (
        10**logphi
        * 10**((logm - logmstar) * (alpha + 1))
        * exp(-10**(logm - logmstar))
    )


def zM_dist(z_min=0.08, z_max=1.52, logm_min=9, logm_max=12.5, **kwargs):
    phytorchx.set_defaults(kwargs)

    zgrid_edges, pz = (
        zprobs := torch.tensor(np.loadtxt(
            priors_beta.file_pdf_of_z_l20
        ).T, **kwargs)
    )[..., torch.logical_and(
        zprobs[0] >= z_min,
        zprobs[0] <= z_max
    )]
    zgrid = phytorchx.mid_one(zgrid_edges)

    logmgrid_edges = torch.linspace(logm_min, logm_max, round((logm_max - logm_min) / 0.01) + 1)
    logmgrid = phytorchx.mid_one(logmgrid_edges)

    pars = {
        key: priors_beta.parameter_at_z0(val, zgrid.clamp_min(0.2)) if len(val) == 3 else val[0]
        for key, val in priors_beta.pars.items()
    }

    return HistDist((logmgrid_edges, zgrid_edges), phytorchx.mid_one(pz).log() + phytorchx.logsumnormalise((
        schechter(logmgrid.unsqueeze(-1), pars['logphi1'], pars['logmstar'], pars['alpha1'])
        +
        schechter(logmgrid.unsqueeze(-1), pars['logphi2'], pars['logmstar'], pars['alpha2'])
    ).log(), 0))


def delta_t_dex(logmass: Tensor):
    return -0.2 + (logmass.clamp(9, 12) - 9) / 3


def expe_logsfr_ratios(zred: Tensor, logmass: Tensor):
    return zred.new_tensor(np.array([
        priors_beta.expe_logsfr_ratios(
            this_z=this_z, this_m=this_m,
            nbins_sfh=7, logsfr_ratio_mini=-5, logsfr_ratio_maxi=5
        )
        for this_z, this_m in tqdm(
            zip(zred.tolist(), logmass.tolist()),
            total=len(zred), leave=False
        )
    ]))


priors = {
    'gas_logz': Uniform(-2, 0.5),

    'dust2': Uniform(0, 4),
    'dust_ratio': conundis.Normal(
        1, 0.3,
        constraint_lower=0, constraint_upper=2
    ),
    'dust_index': Uniform(-1, 0.4),

    'duste_umin': Uniform(0.1, 25),
    'duste_qpah': Uniform(0.5, 7),
    'duste_gamma': TransformedDistribution(
        Uniform(log(0.001), log(0.15)),
        [ExpTransform()]
    ),

    'logfagn': Uniform(log(1e-5), log(3)),
    'logagn_tau': Uniform(log(5), log(150)),
}


class ProspectorSim(PyroModule):
    _sample_params: Iterable[str]

    @staticmethod
    def to_speculator(sim):
        sim['sqrt_dust2'] = sim['dust2']**0.5
        for i, val in enumerate(sim['logsfr_ratios'].unbind(-1), start=1):
            sim[f'logsfr_ratios{i}'] = val
        return sim

    @pyro_method
    def sample(self):
        return {p: getattr(self, p) for p in self._sample_params}



class PASim(ProspectorSim):
    dust2: Tensor
    dust_ratio: Tensor
    dust_index: Tensor

    gas_logz: Tensor
    logfagn: Tensor
    logagn_tau: Tensor

    def __init__(self, nbins=7, z_min=0, z_max=2.5, logm_min=7, logm_max=12.5, **kwargs):
        super().__init__()

        self.nbins = nbins

        self.zred = PyroSample(Uniform(z_min, z_max))
        self.logmass = PyroSample(Uniform(logm_min, logm_max))
        self._logsfr_ratios = PyroSample(StudentT(df=2, loc=0., scale=0.3).expand((self.nbins-1,)).to_event(1))

        for key in type(self).__annotations__:
            setattr(self, key, PyroSample(priors[key]))

    @staticmethod
    def massmet_mean(logmass: Tensor):
        return -0.65 + 0.8 * torch.sigmoid((logmass-10.1) / 0.3)

    @staticmethod
    def massmet_std(logmass: Tensor):
        return 0.52 - 0.36 * torch.sigmoid((logmass-10.4) / 0.2)

    @PyroSample
    def logzsol(self):
        return conundis.Normal(
            self.massmet_mean(self.logmass),
            self.massmet_std(self.logmass),
            constraint_lower=-1.98, constraint_upper=0.19
        )

    @PyroDeterministic
    def logsfr_ratios(self):
        return self._logsfr_ratios.clamp(-5, 5)

    _sample_params = (
        'zred', 'logmass', 'logzsol', 'logsfr_ratios',
        'gas_logz', 'logfagn', 'logagn_tau',
        'dust2', 'dust_index', 'dust_ratio'
    )


class PBSim(ProspectorSim):
    dust_ratio: Tensor

    gas_logz: Tensor

    # duste_min: Tensor
    # duste_qpah: Tensor
    # duste_gamma: Tensor

    logfagn: Tensor
    logagn_tau: Tensor

    def __init__(self, nbins=7, z_min=0.08, z_max=1.52, logm_min=9, logm_max=12.5, **kwargs):
        super().__init__()

        self.nbins = nbins
        self.zM = PyroSample(zM_dist(z_min, z_max, logm_min, logm_max, **kwargs))

        for key in type(self).__annotations__:
            setattr(self, key, PyroSample(priors[key]))

    @partial(PyroDeterministic, event_dim=0)
    def logmass(self) -> Tensor:
        return self.zM[..., 0]

    @partial(PyroDeterministic, event_dim=0)
    def zred(self) -> Tensor:
        return self.zM[..., 1]

    @staticmethod
    def massmet_mean(logmass: Tensor):
        return -0.65 + 0.8 * torch.sigmoid((logmass-10.1) / 0.3)

    @staticmethod
    def massmet_std(logmass: Tensor):
        return 0.52 - 0.36 * torch.sigmoid((logmass-10.4) / 0.2)

    @PyroSample
    def logzsol(self):
        return conundis.Normal(
            self.massmet_mean(self.logmass),
            self.massmet_std(self.logmass),
            constraint_lower=-1.98, constraint_upper=0.19
        )

    @PyroSample
    def _logsfr_ratios(self):
        return StudentT(df=2, loc=expe_logsfr_ratios(self.zred, self.logmass), scale=0.3).to_event(1)

    @PyroDeterministic
    def logsfr_ratios(self):
        return self._logsfr_ratios.clamp(-5, 5)

    @partial(PyroDeterministic, event_dim=2)
    def agebins(self):
        lgtuniv = self.zred.new_tensor(
            priors_beta.cosmo.age(self.zred.numpy(force=True)).to('yr').value
        ).log10().unsqueeze(-1)
        return phytorchx.broadcast_cat((
            lgtuniv.new_tensor([0, log10(30e6)]),
            (8 + (log10(0.85)+lgtuniv - 8) * torch.linspace(0, 1, self.nbins-2, device=lgtuniv.device, dtype=lgtuniv.dtype)),
            lgtuniv
        ), -1)

    @partial(PyroDeterministic, event_dim=1)
    def logages(self):
        return phytorchx.mid_one(self.agebins, -1)

    @partial(PyroDeterministic, event_dim=1)
    def logdt(self):
        return (10**self.agebins).diff(dim=-1).log10()

    @partial(PyroDeterministic, event_dim=1)
    def logmasses(self):
        return self.logmass.unsqueeze(-1) + phytorchx.logsumnormalise(log(10)*(torch.cat((
            self.logsfr_ratios.new_zeros((*self.logsfr_ratios.shape[:-1], 1)),
            (-self.logsfr_ratios).cumsum(-1)
        ), -1) + self.logdt), -1)/log(10)

    @partial(PyroDeterministic, event_dim=1)
    def logcmasses(self):
        return (torch.cat((
            (log(10)*self.logmasses).flip(-1).logcumsumexp(-1).flip(-1).exp()[..., 1:],
            self.logmasses.new_zeros((*self.logmasses.shape[:-1], 1))
        ), -1) + (10**self.logmasses)/2).log10()

    @partial(PyroDeterministic, event_dim=1)
    def logsfrs(self):
        return self.logmasses - self.logdt

    @partial(PyroDeterministic, event_dim=1)
    def logssfrs(self):
        return self.logsfrs - self.logcmasses

    @PyroSample
    def dust2(self):
        return conundis.Normal(
            0.2+0.5*self.logsfrs[..., 0].relu(), 0.2,
            constraint_lower=0, constraint_upper=4
        )

    @PyroSample
    def dust_index(self):
        return conundis.Normal(
            -0.095 + 0.111*self.dust2 - 0.0066*self.dust2**2, 0.4,
            constraint_lower=-1, constraint_upper=0.4
        )

    _sample_params = (
        'zred', 'logmass', 'logzsol', 'logsfr_ratios',
        'logmasses', 'logcmasses', 'agebins', 'logages', 'logdt',
        'gas_logz', 'logfagn', 'logagn_tau',
        'dust2', 'dust_index', 'dust_ratio'
    )
