from libcigars.cli import args

import os
import torch
from more_itertools import collapse
from ruamel import yaml
from time import time
from torch.nn import Sequential

import phytorchx
from phytorchx.vltensor import VLTensor

import clipppy.commands.lightning.hyper as h
from clipppy.commands import NRE
from clipppy.commands.lightning.callbacks import MultiNREPosteriorCallback
from clipppy.commands.lightning.config.schedulers import StepLR
from clipppy.commands.lightning.patches import WandbHooker, Trainer, ModelCheckpoint
from clipppy.sbi._typing import SBIBatch
from clipppy.sbi.data import SBIProcessor
from clipppy.sbi.nn import MultiSBITail
from clipppy.sbi.nn.nre import NRETail
from clipppy.sbi.nn.sets import SetSBIHead, ConditionedSetNRETail, LocalSetNRETail
from clipppy.utils.nn import LazyResidBlock
from clipppy.utils.nn.lpop import LeakyPOP
from clipppy.utils.nn.sets import SetCollapser, Elementwise

from libcigars import CigarsHelper

#%% Distributed
num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
devices_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
num_tasks = num_nodes * devices_per_node

rank, local_rank, world_size = (int(os.environ[key]) for key in ('SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NPROCS'))
print('Distributed:', rank, local_rank, num_nodes, devices_per_node, num_tasks, world_size)

#%% Training technicalities
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')

#%%
c = CigarsHelper.from_args()

#%% Definition of inference groups
STEPPARAMS = 'step_M_logmass', 'step_M_logmass_loc', 'gamma_M_logzsol', 'gamma_M_age'
STEPGROUPS = (STEPPARAMS[0], STEPPARAMS[2]), (STEPPARAMS[0], STEPPARAMS[3]),  (STEPPARAMS[2], STEPPARAMS[3])
GROUPS_GLOBAL = c.cosmogroup, c.dtdgroup, 'M0', 'sigma_res', 'alpha', 'beta', 'alpha_c', STEPPARAMS[1], *STEPGROUPS

GROUPS_LOCAL = 'z', 'logmass', 'logzsol', 'dust_index', 'dust2', 'x_int', 'c_int', 'delta_M'

SLGROUPS = 'delta_M',

nre = NRE(obs_names=('gmags_obs', 'snobs'), param_names=tuple(
    (PARAMS_GLOBAL := set(collapse(GROUPS_GLOBAL))) |
    (PARAMS_LOCAL := set(collapse(GROUPS_LOCAL)))
))

#%% Network and training hyperparameters
kw = dict(bias=False, whiten=dict(affine=False))

hp = h.Hyperparams(
    h.Structure(
        head=h.BaseHParams(
            global_feat=h.OMLP(2, 256, 256, kwargs=kw),
            cds_feat=h.MLP(1, 128, 128, kwargs=kw),
            cds_nresid=5,
            shead=h.MLP(2, 256, kwargs=kw),
        ),
        tail=h.Tail(
            xhead=h.MLP(1, 128, 128, kwargs=kw),
            thead=h.MLP(1, 128, 128, kwargs=kw),
            net=h.OMLP(3, 256, kwargs=kw)
        )
    ),
    h.Training(
        batch_size=64 if c.counts == 1000 else 32,
        lr=h.lrmap.get(args.lr_name, args.lr_step_init),
        scheduler=(
            h.Scheduler(StepLR, step_size=args.lr_step_size, gamma=args.lr_step_gamma)
            if args.lr_name == 'step' else None
        )
    )
)

MAX_BATCH_SIZE = (32 if c.counts == 1000 else 4) * world_size

memory_batch_size, accumulate_grad_batches = nre.set_training(hp, MAX_BATCH_SIZE)
memory_batch_size = memory_batch_size // num_tasks

nre.just_save_hyperparameters(dict(hp.collapse()))

#%% Load target data and training/validation sets
data = c.mock_data
datamod = c.datamod(
    nre, memory_batch_size, global_names=PARAMS_GLOBAL,
    ddp_val=(c.counts > 1000 and num_tasks > 1),
    preload=c.counts == 1000)

#%% Define network
hps = hp.structure

norms = c.norms

# Deterministic weights initialisation across DDP!
torch.manual_seed(42)

nre.head = SetSBIHead(
    obs_pre=norms,
    whiten=False,
    head=Sequential(Elementwise(hps.head.global_feat.make())),
    event_dims={'gmags_obs': 1, 'snobs': 1}, obs_names=nre.obs_names)
nre.tail = MultiSBITail(tails={
    group: ConditionedSetNRETail(
        head=SetCollapser(Elementwise(Sequential(
            hps.head.cds_feat.make(),
            *(LazyResidBlock(**kw) for _ in range(hps.head.cds_nresid))
        )), lens_scale=c.counts / 1000 if c.counts != 1000 else None),
        tail=NRETail(thead=hps.tail.thead.make(), xhead=hps.tail.xhead.make(),
                     net=Sequential(hps.tail.net.make(), LeakyPOP())),
        set_norm=True,
        params_pre=norms
    )
    for group in GROUPS_GLOBAL
} | {
    group: LocalSetNRETail(
        subsample=100,
        net=torch.nn.Sequential(hps.tail.net.make(), LeakyPOP()),
        summarize=summ, shead=(SetCollapser(Elementwise(hps.head.shead.make())) if summ else None),
        params_pre=norms
    ) for group in GROUPS_LOCAL for summ in [group in SLGROUPS]
})

#%% Load base network to fine-tune, if requested
fine_tune = yaml.YAML(typ='safe', pure=True).load((c.base_resdir / 'finetune.yaml').open()).get(c.name)
# fine_tune = False

if args.fine_tune and fine_tune:
    from libcigars.utils import load_for_finetuning

    fine_tune_name = c.clone(**fine_tune).bestnet_ckpt
    print(f'Fine-tuning from {fine_tune_name!s}')
    load_for_finetuning(nre, fine_tune_name)

#%% Sanity check / init NN
print('Initialising and sanity check:')
ex = next(iter(datamod.val_dataset.batched(memory_batch_size, False)))
for key in list(ex.keys()):
    if isinstance(ex[key], list) and isinstance(ex[key][0], VLTensor):
        ex[key] = torch.utils.data.default_collate(ex[key])
ex = SBIBatch(*SBIProcessor(param_names=nre.param_names, obs_names=nre.obs_names).split(ex))
print(nre.train().to(phytorchx.get_default_device())(ex))

#%% Train...
torch.manual_seed(int(time()) + 100*os.getpid())

import warnings

warnings.filterwarnings('ignore', 'elementwise comparison failed', FutureWarning)
warnings.filterwarnings('ignore', 'Trying to infer the `batch_size` from an ambiguous collection.')


trainer = Trainer(
    max_steps=args.max_steps,

    accelerator='gpu', strategy='ddp',
    num_nodes=num_nodes, devices=devices_per_node,

    logger=c.logger,
    accumulate_grad_batches=accumulate_grad_batches,
    callbacks=[
        WandbHooker(args.wandb_project, name=c.logger.name, username=args.wandb_user, kwargs=dict(mode=args.wandb_mode)),
        *((MultiNREPosteriorCallback(
            groups_global=GROUPS_GLOBAL,
            data={key: [data[key].as_subclass(torch.Tensor).to('cuda')] for key in nre.obs_names},
            plotter=datamod.posterior_plotter.to('cuda').copy(truths=data), device='cuda',
            on_validation=True
        ),) if rank == 0 else ()),
        ModelCheckpoint()
    ],

    val_check_interval=1000, check_val_every_n_epoch=None,
)

trainer.fit(nre, datamod)
