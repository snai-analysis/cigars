from libcigars.cli import args

from contextlib import nullcontext
from itertools import repeat, chain

import torch
from torch import Tensor
from tqdm.auto import tqdm

if torch.cuda.is_available():
    torch.set_default_device('cuda')

from phytorchx.vltensor import VLTensor
from clipppy.contrib.ncdatagen import NCDatagen
from clipppy.distributions.conundis import ConstrainingMessenger
from clipppy.distributions.simplex.simplifying_messenger import SimplifyingMessenger
from clipppy.sbi.persistent.netcdf_data import NetCDFDataFrame
from clipppy.utils.nn import MultiModule, LazyWhitenOnline

from libcigars import CigarsHelper

#%%
c = CigarsHelper.from_args()
model = c.model

#%% Set up variables to save
snobs_obs = 'm_obs', 'x_obs', 'c_obs'
snobs_Ws = 'Wmm', 'Wxx', 'Wcc'

keys_global, keys_local = set(model.global_vars), set(model.local_vars) - {'gmags', 'x_ext', 'c_ext', 'E'}
snobs_keys = snobs_obs + snobs_Ws


def sim_one(ctx=nullcontext(), **kwargs):
    with ctx:
        sim = model.generate(keys_global, keys_local, **kwargs)
        return sim | {'snobs': torch.stack([sim.pop(key) for key in snobs_keys], -1)}


print(f'keys_global = {keys_global!r}')
print(f'keys_local = {keys_local!r}')
print(f'snobs_keys = {snobs_keys!r}')

#%% Load prior constraints, if available
if c.cp_name.is_file():
    cp = c.cp

    for ctx in cp.ctxs:
        if isinstance(ctx, ConstrainingMessenger):
            ctx.ranges = {model.find_var(key): val for key, val in ctx.ranges.items()}
        if isinstance(ctx, SimplifyingMessenger):
            ctx.dists = {
                model.find_var(key): (tuple(model.find_var(k) for k in group), val)
                for key, (group, val) in ctx.dists.items()
            }
else:
    cp = nullcontext()


print('cp:', cp)

#%% Generate training and validation sets
NCDatagen(
    64_000, 64_00, f'{c.traindir!s}/{{suffix}}.nc',
    lambda: {
        key: ([val] if isinstance(val, VLTensor) else val.unsqueeze(0))
        for key, val in sim_one(cp).items()
    },
    var_dimensions={'gmags_obs': ('mags',), 'snobs': ('snobs',)},
    dimlens={'mags': 6, 'snobs': len(snobs_keys)}
).run()

#%% Calculate input normalisations, if requested
if args.norms:
    print('Whitening')

    whiteners = MultiModule({
        key: LazyWhitenOnline(ndim) for key, ndim in chain(
            zip((*keys_global, *(keys_local-set(snobs_keys)-{'gmags_obs', 'snobs'})), repeat(0)),
            zip(('gmags_obs', 'snobs'), repeat(1)),
        )
    })

    for ex in tqdm(NetCDFDataFrame(f'{c.traindir!s}/train.nc', keys=whiteners.mods.keys())):
        for key, val in ex.items():
            whiteners.mods[key](val.as_subclass(Tensor))

    for mod in whiteners.mods.values():
        mod.freeze_()
    c.norms = whiteners
