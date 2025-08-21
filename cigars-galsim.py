import pyro
import torch
from tqdm.auto import tqdm

from phytorchx.dataframe import TensorDataFrame

if torch.cuda.is_available():
    from clipppy.patches import torch_numpy
    torch.set_default_device('cuda')


from libcigars.galaxy import Galaxy
from libcigars.galaxy.prospector import PBSim


#%% Generate parameter hierarchy
pro = PBSim(z_min=0.02, z_max=1.5, logm_min=8.5, logm_max=12.5)
with pyro.plate('plate', 1000_000):
    sims = TensorDataFrame(pro.sample())


#%% Generate redshifted absolute magnitudes
g = Galaxy()
sims.data['M'] = torch.cat([
    g.generate(pro.to_speculator(p)) for p in tqdm(sims.batched(100, shuffle=False))
], 0)


#%% Save for later
torch.save(sims, 'train/prospector-beta-sims.pt')
