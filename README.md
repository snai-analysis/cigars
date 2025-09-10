# <img src="https://emojis.slackmojis.com/emojis/images/1702006970/80724/bender_cigarq.png" style="height: 1em" /> CIGaRS I

This repository contains the code and outputs from

> "Combined simulation-based inference from SNæ Ia and host photometry"
>
> by Konstantin Karchev, Roberto Trotta, and Raul Jimenez
>
> [![arXiv:2508.XXXXX](https://img.shields.io/badge/arXiv-2508.15899-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2508.15899)

---

Large files (indicated by :package:) are managed with [DVC](https://dvc.org/) and hosted on [DagsHub](https://dagshub.com/kosiokarchev/cigars). They can be downloaded with
```shell
dvc pull
```
from within the root directory of the repo.

# Installation

Pre-requisites for running the code or loading/using the released outputs can be quickly and automatically installed with
```shell
pip install -r requirements.txt
```
We make use of the [Clipppy](https://github.com/kosiokarchev/clipppy), [φtorch](https://github.com/kosiokarchev/phytorch), and [SLICsim](https://github.com/kosiokarchev/slicsim) open-source packages, whose dependencies (refer to the respective repositories) will also be installed. It is advisable to manually install the appropriate version of [PyTorch](https://pytorch.org/) for your hardware in advance.

# Workflow

> [!NOTE]
> This code, especially the network training in step 3 below, is intended to be run on a high-performance computing cluster of many nodes and GPUs using [SLURM](https://slurm.schedmd.com/), whose setup is not documented here. Furthermore, it uses, by default, the [W&B](https://wandb.ai/) experiment-tracking platform, which also requires explicit setup and authentication (and an active connection to the internet during training).

The workflow starts with
1. `cigars-galsim.py` to generate the bank of potential hosts and their (absolute, noiseless) photometry
   - → [`train/prospector-beta-sims.pt`](train/prospector-beta-sims.pt.dvc) :package:.

Then, for given `COUNTS`[^counts] and `ZOOM_STAGE` (starting at zero):

2. `cigars-traingen.py COUNTS ZOOM_STAGE` to generate training and validation data:
   - → `train/cigars/cigars-dindep-COUNTS/ZOOM_STAGE/train.nc`[^train],
   - → [`train/cigars/cigars-dindep-COUNTS/ZOOM_STAGE/val.nc`](train/cigars) :package:,

   and the mock target example (only if it does not exist)
   - → [`data/cigars-dindep-COUNTS.pt`](data/);
3. `cigars-nre.py COUNTS ZOOM_STAGE` to train the network and save checkpoints:[^version]
   - → `lightning_logs/cigars-dindep-COUNTS-ZOOM_STAGE/version_N/checkpoints/*.ckpt`;
4. `cigars-nre-eval.py --version=N COUNTS ZOOM_STAGE` to evaluate the trained network:
   - → `lightning_logs/cigars-dindep-COUNTS-ZOOM_STAGE/version_N/wgplotter.pt`: posteriors for global parameters (weighted samples),
   - → `lightning_logs/cigars-dindep-COUNTS-ZOOM_STAGE/version_N/wlplotter.pt`: posteriors for local parameters (weighted samples),
   - → `lightning_logs/cigars-dindep-COUNTS-ZOOM_STAGE/version_N/lpoststats.pt`: posterior moments for local parameters;

Then, optionally,

5. `cigars-constrain.py COUNTS ZOOM_STAGE` to calculate bounds useful for prior truncation:
   - → [`train/cigars/cigars-dindep-COUNTS/ZOOM_STAGE+1/cp.pt`](train/cigars/)

   (note that this step requires editing the [`res/bounds.yaml`](res/bounds.yaml) file manually). Then repeat steps 2–4 above with `ZOOM_STAGE+1` until convergence. Fine-tuning is controlled by the [`res/finetune.yaml`](res/finetune.yaml) file that lists the "source" of fine-tuning in every stage (notice that we also fine-tune across an increase in counts).

# Results

We release the "best" trained network from each stage (based on validation loss averaged across inferred parameters) in
- [`res/cigars-dindep-COUNTS/ZOOM_STAGE/bestnet.pt`](res/) :package:

and the final results[^plotters] used for creating the plots
- [`res/cigars-dindep-COUNTS/ZOOM_STAGE/wgplotter.pt`](res/),
- [`res/cigars-dindep-COUNTS/ZOOM_STAGE/lpoststats.pt`](res/).

[^counts]: `COUNTS` is a *label* for the size of the analysed/mocked survey. It was tuned based on the size of the galaxy bank (1000000) under the assumption of matching host- and SN dust extinction. With the independent-dust model we use in the paper (labelled "`dindep`" in file names), the two examples—with roughly 1600 and 16000 objects—correspond to `COUNTS` of `1000` and `10000`.
[^train]: The training data sets are not included in the repository because of their size. They are simply a 10x larger version of the validation sets (which *are* released) and can be re-generated as detailed here.
[^version]: This can be repeated and will result in a new version `N`. We ran multiple versions while developing the analysis, but one run with the final architecture and settings should be enough to reproduce our results.
[^plotters]: Note that the results for each global-parameter group (in `wgplotter`s) are derived from a generally different checkpoint based on the individual respective validation loss, while the local-parameter results all come from the same `bestnet`. We do not release the raw `wlplotter`s due to their size.