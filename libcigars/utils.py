from pathlib import Path

import torch
from more_itertools import collapse
from tqdm.auto import tqdm

import phytorchx
from clipppy.sbi.nn.sets import ConditionedSetNRETail
from clipppy.utils.nn import LazyWhitenOnline
from phytorchx.dataframe import TensorDataFrame
from clipppy.utils.plotting.sbi import MultiSBIPosteriorPlotter


def load_for_finetuning(nre, ckpt, reset_frac=float('inf')):
    from clipppy.utils.nn import WhitenOnline
    from clipppy.sbi.nn import ModuleDict2
    from clipppy.sbi.nn.sets import LocalSetNRETail

    nre.head, tail = phytorchx.load(ckpt)['clipppy_nets']

    if not isinstance(tail.tails, ModuleDict2):
        tail.tails = ModuleDict2(tail.tails.items())

    replaced = {nre.head}
    for key, old in nre.tail.tails.items():
        if key in tail.tails:
            print('Fine-tuning tail', key)
            nre.tail.tails[key] = new = tail.tails[key]
            replaced.add(new)

            if isinstance(new, ConditionedSetNRETail):
                new.head.lens_scale = old.head.lens_scale

    for rm in replaced:
        for m in rm.modules():
            if isinstance(m, WhitenOnline) and not isinstance(m, LazyWhitenOnline):
                m.freeze_()
            # m.n = int(m.n // reset_frac)


def load_nre(path):
    from clipppy.commands.lightning.nre import NRE
    from clipppy.sbi.nn.sets import LocalSetNRETail

    nre = NRE()
    nre.head, nre.tail = phytorchx.load(path)['clipppy_nets']
    nre.eval()

    groups_global = tuple(key for key, val in nre.tail.tails.items() if not isinstance(val, LocalSetNRETail))
    params_global = set(collapse(groups_global))
    groups_local = tuple(key for key, val in nre.tail.tails.items() if isinstance(val, LocalSetNRETail))
    params_local = set(collapse(groups_local))

    for key in params_local:
        nre.tail.tails[key].subsampling = False

    nre.param_names = *params_global, *params_local
    nre.obs_names = nre.head.obs_names

    return nre, groups_global, groups_local


def get_best_global_results(logdir: Path, gplotter: MultiSBIPosteriorPlotter, data, obs=None) -> MultiSBIPosteriorPlotter:
    if obs is None:
        obs = data

    import tensorboard.backend.event_processing.event_accumulator as ea

    tb = ea.EventAccumulator(str(logdir), {ea.SCALARS: 0}).Reload()

    lws = {}
    for tag in tb.Tags()['scalars']:
        if not tag.startswith('val/'):
            continue

        key = tag[4:]
        if key.startswith('('):
            key = eval(key)

        beststep = min(tb.Scalars(tag), key=lambda s: s.value).step + 1
        nre, groups_global, *_ = load_nre(logdir / f'checkpoints/step={beststep}.ckpt')

        if key in groups_global:
            print(key, beststep)
            lws.update(gplotter._eval_nre((key,), nre, gplotter._samples, obs))

    return gplotter.with_ratios(lws, truths=data)


def eval_nre_local(nre, params, obs, names, batch_size):
    lnr = {key: [] for key in names}
    with torch.inference_mode():
        for params in tqdm(TensorDataFrame(params).batched(batch_size, shuffle=False), leave=False):
            for key, val in lnr.items():
                _params, _obs = nre.head(params, obs)
                val.append(nre.tail.forward_one(key, {key: p[..., None] for key, p in _params.items()}, _obs).squeeze(0))
    return {key: val - val.logsumexp(0, keepdim=True) for key, val in lnr.items() for val in [torch.cat(val, 0)]}
