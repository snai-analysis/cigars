# <img src="https://emojis.slackmojis.com/emojis/images/1702006970/80724/bender_cigarq.png" style="height: 1em" /> CIGaRS I

This repository contains the code and outputs from

> "Combined simulation-based inference from SNæ Ia and host photometry"
> 
> by Konstantin Karchev, Roberto Trotta, and Raul Jimenez
> 
> [![arXiv:2508.XXXXX](https://img.shields.io/badge/arXiv-2508.15899-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2508.15899)

Large files (simulations and trained networks) are managed with [DVC](https://dvc.org/) and hosted on [DagsHub](https://dagshub.com/kosiokarchev/cigars).

---

Pre-requisites for running the code can be installed with
```shell
pip install -r requirements.txt
```

---

The workflow follows
1. `cigars-galsim.py` to generate the bank of potential hosts and their (absolute, noiseless) photometry
   - → `train/prospector-beta-sims.py`;
2. `cigars-traingen.py` to generate training and validation data:
   - →`train/cigars/*/(train|val).nc`
   
   (and the mock target example →`data/*.pt`, if it does not exist);
3. `cigars-nre.py` to train the network and save checkpoints:
   - →`lightning_logs/*/checkpoints/*.ckpt`;
4. `cigars-nre-eval.py` to evaluate the trained network:
   - → `lightning_logs/*/wgplotter.pt`: posteriors for global parameters (weighted samples),
   - → `lightning_logs/*/wlplotter.pt`: posteriors for local parameters (weighted samples),
   - → `lightning_logs/*/lpoststats.pt`: posterior moments for local parameters.

Then, optionally,
5. `cigars-constrain.py` to calculate bounds useful for prior truncation:
   - → `train/cigars/*/cp.pt`,

   and repreat 2–4 above until convergence.

Parameters `COUNTS ZOOM_STAGE` must be given to the scripts 2–5 above (see the `--help` and `libcigars/cli.py` for more command-line arguments). Particularly, the `COUNTS` is a *label* for the size of the analysed survey, with the two examples in the paper (with roughly 1600 anf 16000 objects) corresponding to `COUNTS` of `1000` and `10000`.
