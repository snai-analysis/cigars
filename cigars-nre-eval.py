from libcigars.cli import args

import torch, phytorchx
from more_itertools import collapse

if torch.cuda.is_available():
    from clipppy.patches import torch_numpy
    torch.set_default_device('cuda')
    torch.set_float32_matmul_precision('high')

from clipppy.commands import NRE
from clipppy.utils.plotting.sbi import MultiSBIPosteriorPlotter

from libcigars import CigarsHelper

c = CigarsHelper.from_args()
data = c.mock_data
obs = {key: [data[key].as_subclass(torch.Tensor)] for key in ('gmags_obs', 'snobs')}

logdir = c.logdir / f'version_{args.version}'

#%% GLOBAL
from libcigars.utils import get_best_global_results

datamod = c.datamod(NRE(
    param_names=tuple(c.model.global_vars),
    obs_names=('gmags_obs', 'snobs')
), 64, global_names=set(c.model.global_vars), preload=False)

gplotter = datamod.posterior_plotter
wgplotter = get_best_global_results(logdir, gplotter, data, obs)

torch.save(wgplotter, logdir / 'wgplotter.pt')

#%% LOCAL
from clipppy.commands.lightning.utils import get_best_ckpt
from libcigars.utils import eval_nre_local

nre = NRE()

ckpt = get_best_ckpt(logdir)
print(ckpt)
nre.head, nre.tail = phytorchx.load(ckpt)['clipppy_nets']

nre.obs_names = nre.head.obs_names
nre.param_names = tuple(collapse(nre.tail.tails.keys()))

from clipppy.sbi.nn.sets import LocalSetNRETail
for mod in nre.modules():
    if isinstance(mod, LocalSetNRETail):
        mod.subsampling = False


datamod = c.datamod(
    nre, 64, preload=True,
    global_names=set(nre.param_names) & set(c.model.global_vars),
    local_names=set(nre.param_names) & set(c.model.local_vars))


wlplotter = MultiSBIPosteriorPlotter(datamod.local_val_params, truths=data, labels=c.labels).with_ratios(
    eval_nre_local(nre, datamod.local_val_params, obs, datamod.local_names, 64))
lpoststats = {param: {
    'mean': torch.from_numpy(stats.means[param].to_numpy()),
    'std': torch.from_numpy(stats.stds[param].to_numpy())
} for param in datamod.local_val_params.keys() for stats in [wlplotter.stats(param)]}

torch.save(wlplotter, logdir / 'wlplotter.pt')
torch.save(lpoststats, logdir / 'lpoststats.pt')
