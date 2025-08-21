from __future__ import annotations

from functools import partial
from math import log
from typing import Callable, Union, Annotated, get_origin, get_args, Mapping, TYPE_CHECKING

import astropy.units as u
import torch
from astropy.cosmology import WMAP9, z_at_value
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Sequential

if TYPE_CHECKING:
    from speculator import Speculator


def spec_layer(spec: Speculator, i: int):
    w, b = spec.W_[i], spec.b_[i]
    ret = Linear(w.shape[0], w.shape[1])
    ret.weight = Parameter(torch.tensor(w).T)
    ret.bias = Parameter(torch.tensor(b))
    return Sequential(ret, JAct(
        torch.tensor(spec.alphas[i].numpy(), dtype=torch.get_default_dtype()),
        torch.tensor(spec.betas[i].numpy(), dtype=torch.get_default_dtype())
    )) if i < spec.n_layers-1 else ret


class JAct(Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.malpha = Parameter(-alpha)
        self.beta = Parameter(beta)

    def forward(self, x: Tensor):
        # return (1-self.beta).div_((self.malpha*x).exp_().add_(1)).add_(self.beta).mul_(x)
        return (self.malpha*x).exp_().add_(1).reciprocal_().mul_(1-self.beta).add_(self.beta).mul_(x)


class TorchSpeculator(Module):
    parameters_shift: Annotated[Tensor, None]
    parameters_scale: Annotated[Tensor, None]
    pca_shift: Annotated[Tensor, None]
    pca_scale: Annotated[Tensor, None]
    pca_transform_matrix: Annotated[Tensor, None]
    log_spectrum_scale: Annotated[Tensor, None]
    log_spectrum_shift: Annotated[Tensor, None]

    wavelengths: Annotated[Tensor, 'wavelengths']
    layers: Union[Callable[[Tensor], Tensor], Module]

    def from_spec(self, spec):
        for name, ann in type(self).__annotations__.items():
            if get_origin(ann) is Annotated:
                self.register_buffer(
                    name, torch.tensor(getattr(spec, get_args(ann)[1] or f'{name}_'), dtype=torch.get_default_dtype())
                )

        self.layers = Sequential(*map(partial(spec_layer, spec), range(spec.n_layers)))

        return self

    def forward(self, x):
        return (
            (
                self.layers(
                    (x-self.parameters_shift)/self.parameters_scale
                ) * self.pca_scale + self.pca_shift
            ) @ self.pca_transform_matrix
        ) * self.log_spectrum_scale + self.log_spectrum_shift


class SpeculatorAlpha(Module):
    param_names = 'logzsol', *(f'logsfr_ratios{i+1}' for i in range(6)), 'sqrt_dust2', 'dust_index', 'dust_ratio', 'logfagn', 'logagn_tau', 'gas_logz', 'zred'

    wavelengths: Tensor

    def __init__(self, subs):
        super().__init__()
        self.subs = subs
        self.register_buffer('wavelengths', torch.cat([
            s.wavelengths for s in subs
        ], -1))

    @classmethod
    def beta_to_params(cls, **kwargs) -> dict[str, float]:
        kwargs['logfagn'] = log(kwargs['fagn'])
        kwargs['logagn_tau'] = log(kwargs['agn_tau'])
        for i, val in enumerate(kwargs['logsfr_ratios'], start=1):
            kwargs[f'logsfr_ratios{i}'] = val
        kwargs['sqrt_dust2'] = kwargs['dust2']**0.5
        return {key: float(kwargs[key]) for key in cls.param_names}

    @staticmethod
    def tage_to_zred(tage: float):
        return float(z_at_value(WMAP9.age, tage*u.Gyr))

    def forward(self, params: Mapping[str, Tensor]):
        x = torch.stack([params[p] for p in self.param_names], -1)
        return torch.cat([s(x) for s in self.subs], -1)
