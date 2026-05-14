# <img src="https://emojis.slackmojis.com/emojis/images/1702006970/80724/bender_cigarq.png" style="height: 1em" /> CIGaRS I

This repository contains the code and outputs from

> "CIGaRS I: combined simulation-based inference from type Ia supernovae and host photometry"
>
> Konstantin Karchev, Roberto Trotta, and Raúl Jiménez. *Nature Astronomy* (2026).
>
>
> [![arXiv:2508.15899](https://img.shields.io/badge/arXiv-2508.15899-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2508.15899)
> [![DOI:10.1038/s41550-026-02842-5](https://img.shields.io/badge/DOI-10.1038%2Fs41550--026--02842--5-b31b1b?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAQAAAD9CzEMAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAD/h4/MvwAAAAlwSFlzAAAXEQAAFxEByibzPwAAAAd0SU1FB+gDGgg5BT461ukAAABedEVYdFJhdyBwcm9maWxlIHR5cGUgaXB0YwAKaXB0YwogICAgICAyOAozODQyNDk0ZDA0MDQwMDAwMDAwMDAwMGYxYzAyNmUwMDAzNTI0NjQ3MWMwMjAwMDAwMjAwMDQwMApgmj2eAAADBUlEQVRYw72Yv09UQRDHP2857jQoYIgQ5Ydaef7EgggFvRGjhZ0awx9gYkFlc4maWBgbekzoTKwwJkJDvFiYSEJDvCCHv8JdTK4h5s4YRY/7WpzoW96+uwN8zlbz5u13ZnZmZ2fXox61cpxBBkjSSxsJYI0ieZaYZ45FSrWnezVkhqOMcJ6TdBBzyMuskmGGabJU6hoaoH7GWUENjBXG6d8aeBcpcg2Bb4wcKboahR8mTWVL8EJUSDNcH7yJ0S3abvsxSlMt+GbGKG4bXogiYzSHWz/G1x3BC/GVsTAvRndo/V8vRt2h3f7aB2MRCHcX6VpT4or7uL3q1mEd0J7wGenNSZsKS8x2ndVNPdY5IWQ0qPt6oaw+6I2e655OhyVtyt61uaDNh3RJD/RSq5Kka0LtuquCbHqvy2HL9Gd3G8aD8Lf1Tt//wJR1RS16qIqC9FYn3CrGMVUFx1w157rWfCBlXdMtrctNd9wKVjgGBrhAXzCpnvLKx1U4ww1EgSzLfA4kYIsrL/sYAWhl1h3elOXBsp7pqo7roLo1pEeWN1n1uH2YpRWGKLgVXPTFQHqifT5ZnzI+2Scl3QoKDBkG6XBv7ALffFzeWphPvLZqTEht6GDQMOA8rYAflH2cffSt84UGKMaAIdnIn0FSY78lDb3bU9Ag9RraIlXQZkhEqiBhIoUHDGuR4q8ZipEqKBrykSrIG5Ya+9Orw4fQkmHe2rA+arJA7GLgWbwJKxVl5g1zrLqlsU0gNvn5prBqs8qcYZGMW7rbmtZs+eMRt0zZ5YbIsGgoMeOWtlkgLZY/MeuIiYeVgxlKBpgm55J2Wj3gPktdnHaL63QB5JiuLmWWKZf8iLUonZbNe9jv4wyHXQBTZKsKKkwGd0OCUxbfQ4+P66Xbkp4Odrx5JqnwO44LTHDHTu0EGUq+ql+28sbjiS8mHh9J8NM/XUyw4P8QaB09mU3Dqymt1zpG3vxC5O37f7iARH6FqnoR6SVwI9wRXmM3kjbSi3iV/tFTQuSPIfVPvh0+5/wCZHf95+bZniQAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjQtMDMtMjZUMDg6NTY6NTUrMDA6MDBIVKTTAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI0LTAzLTI2VDA4OjU2OjU1KzAwOjAwOQkcbwAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNC0wMy0yNlQwODo1NzowNSswMDowMMk+WOoAAAAASUVORK5CYII=)](https://doi.org/10.1038/s41550-026-02842-5)
> [![zenodo.18705765](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18705765-b31b1b?logo=zenodo)](https://doi.org/10.5281/zenodo.18705765)

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
> Tested on Linux and Windows using Python 3.12 and PyTorch 2.6, although earlier versions should work as well, as long as the environment is consistent.

We release several Python scripts (see [`libcigars/cli.py`](libcigars/cli.py) for the command-line options (not all apply to a given script)) corresponding to different aspects of our analysis. The workflow starts with
1. [`cigars-galsim.py`](cigars-galsim.py) to generate the bank[^bank] of potential hosts and their (absolute, noiseless) photometry
   - → [`train/prospector-beta-sims.pt`](train/prospector-beta-sims.pt.dvc) :package:.

Then, for given `COUNTS`[^counts] and `ZOOM_STAGE` (starting at zero):

2. [`cigars-traingen.py`](cigars-traingen.py)` COUNTS ZOOM_STAGE` to generate the mock target example (only if it does not exist)
   - → [`data/cigars-dindep-COUNTS.pt`](data/)

   and training and validation data:
   - → `train/cigars/cigars-dindep-COUNTS/ZOOM_STAGE/train.nc`[^train],
   - → [`train/cigars/cigars-dindep-COUNTS/ZOOM_STAGE/val.nc`](train/cigars) :package:;

3. [`cigars-nre.py`](cigars-nre.py)` COUNTS ZOOM_STAGE` to train the network and save checkpoints;

   This step is intended to be run on a high-performance computing cluster of many nodes and GPUs (we used up to 8 NVIDIA A100 with 64 GB memory each) using [SLURM](https://slurm.schedmd.com/), whose setup is not documented here. Furthermore, it uses, by default, the [W&B](https://wandb.ai/) experiment-tracking platform, which requires explicit setup and authentication (and an active connection to the internet during training).
4. [`cigars-nre-eval.py`](cigars-nre-eval.py)` --version=N COUNTS ZOOM_STAGE` to evaluate the trained network.

Then, optionally,

5. [`cigars-constrain.py`](cigars-constrain.py)` COUNTS ZOOM_STAGE` to calculate bounds useful for prior truncation (requires editing the [`res/bounds.yaml`](res/bounds.yaml) file manually):
   - → [`train/cigars/cigars-dindep-COUNTS/ZOOM_STAGE+1/cp.pt`](train/cigars/).

   Then repeat steps 2–4 above with `ZOOM_STAGE+1` until convergence. Fine-tuning is controlled by the [`res/finetune.yaml`](res/finetune.yaml) file that lists the "source" of fine-tuning in every stage (notice that we also fine-tune across an increase in counts).

# Results

We release the "best" trained network from each stage (the checkpoint with the lowest validation loss averaged across all inferred parameters) in
- [`res/cigars-dindep-COUNTS/ZOOM_STAGE/bestnet.pt`](res/) :package:

and the final results[^plotters] used for creating the plots
- [`res/cigars-dindep-COUNTS/ZOOM_STAGE/wgplotter.pt`](res/): posteriors for global parameters (weighted samples),
- [`res/cigars-dindep-COUNTS/ZOOM_STAGE/lpoststats.pt`](res/): posterior moments (mean and st. dev.) for object-specific (local) parameters (only for `COUNTS=1000`, i.e. the smaller data set).

Finally, we provide an example notebook ([`cigars-plot.ipynb`](cigars-plot.ipynb)) that briefly demonstrates the contents and usage of the result files.

[^bank]: Note that the present bank-generation code does not include the re-weighting procedure described in the Implementation details in the paper since it was added later during review. Please use the provided bank(s) instead.
[^counts]: `COUNTS` is a *label* for the size of the analysed/mocked survey. It was tuned based on the size of the galaxy bank (1000000) under the assumption of matching host- and SN dust extinction. With the independent-dust model we use in the paper (labelled "`dindep`" in file names), the two examples—with roughly 1600 and 16000 objects—correspond to `COUNTS` of `1000` and `10000`.
[^train]: The training data sets are not included in the repository because of their size. They are simply a 10x larger version of the validation sets (which *are* released) and can be re-generated as detailed here.
[^plotters]: Note that the results for each global-parameter group (in `wgplotter`s) are derived from a generally different checkpoint based on the individual respective validation loss, while the local-parameter results all come from the same `bestnet`. We do not release the raw `wlplotter`s due to their size.
