from libcigars.cli import args

import torch
from pyro.distributions import Rejector, Uniform, Normal

import phytorchx
from phytorch.geom import Polygon
from clipppy.distributions.conundis import ConstrainingMessenger
from clipppy.distributions.simplex.simplifying_messenger import SimplifyingMessenger
from clipppy.utils.messengers import MultiContext
from uplot.contour import kdecontour

from libcigars import CigarsHelper

c = CigarsHelper.from_args()

wgplotter = phytorchx.load(c.resdir / 'wgplotter.pt')

#%%
# Print 1D constraints from all groups
# These then need to be added to res/bounds.yaml BY HAND!
for group in wgplotter.weights.keys():
    for p, (bl, bh) in wgplotter.bounds(group, args.thresh).items():
        print(f'{p}: [{bl.item():.4g}, {bh.item():.4g}] ({(bh.item()-bl.item())/(wgplotter.ranges[p][1].item()-wgplotter.ranges[p][0].item()):.0%})')

input('Edit res/bounds.yaml as desired and press ENTER.')

# Load 1D bounds from res/bounds.yaml
cm = ConstrainingMessenger(ranges=c.zoom().bounds)

#%% Add 2D-contour
sm_rej = SimplifyingMessenger({
    group: Rejector(prior, Polygon(_[0], _[1:]).log_indicator, 0)
    for group, prior in (
        (c.cosmogroup, Uniform(torch.tensor([0., 0.]), torch.tensor([1., 1.])).to_event(1)),
        (c.dtdgroup, Normal(torch.tensor([-12.15, -1.34]), torch.tensor([0.1, 0.2])).to_event(1)),
        # (('step_M_logmass', 'gamma_M_logzsol'), Uniform(torch.tensor([-0.2, -0.4]), torch.tensor([0.2, 0.4])).to_event(1))
    )
    for _ in [tuple(map(torch.tensor, kdecontour(wgplotter._samples[group[0]], wgplotter._samples[group[1]], (1-args.thresh,), wgplotter.weights[group])[1-args.thresh]))]
})

#%% Combine and save
c.zoom().cp = MultiContext((cm, sm_rej))
