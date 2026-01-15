from libcigars.cli import args

import torch

if torch.cuda.is_available():
    from clipppy.patches import torch_numpy
    torch.set_default_device('cuda')
    torch.set_float32_matmul_precision('high')

from clipppy.utils.plotting.sbi import MultiSBIPosteriorPlotter

from libcigars import CigarsHelper

c = CigarsHelper.from_args()
data = c.mock_data
obs = {key: [data[key].as_subclass(torch.Tensor)] for key in ('gmags_obs', 'snobs')}

logdir = c.logdir / f'version_{args.version}'

#%%
from clipppy.commands.lightning.utils import get_best_ckpt
from libcigars.utils import load_nre, get_best_results

_nre, *_ = load_nre(get_best_ckpt(logdir))
datamod = c.datamod(
    _nre, 64, preload=True,
    global_names=set(_nre.param_names) & set(c.model.global_vars),
    local_names=set(_nre.param_names) & set(c.model.local_vars))

global_lws, local_lws = get_best_results(logdir, datamod, obs)

#%% GLOBAL
if global_lws:
    wgplotter = MultiSBIPosteriorPlotter(datamod.global_val_params, truths=data, labels=c.labels).with_ratios(global_lws)
    torch.save(wgplotter, logdir / 'wgplotter.pt')

#%% LOCAL
if local_lws:
    wlplotter = MultiSBIPosteriorPlotter(datamod.local_val_params, truths=data, labels=c.labels).with_ratios(local_lws)
    lpoststats = {param: {
        'mean': torch.from_numpy(stats.means[param].to_numpy()),
        'std': torch.from_numpy(stats.stds[param].to_numpy())
    } for param in datamod.local_val_params.keys() for stats in [wlplotter.stats(param)]}

    torch.save(wlplotter, logdir / 'wlplotter.pt')
    torch.save(lpoststats, logdir / 'lpoststats.pt')
