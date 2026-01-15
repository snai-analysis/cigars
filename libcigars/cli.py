import argparse
from typing import Literal


class Args(argparse.Namespace):
    variant: Literal['dindep', 'dglob', 'dloc'] = 'dindep'

    counts: int
    zoom_stage: int

    norms: bool = True

    fine_tune: bool = True
    train_locals: bool = True

    max_steps: int = 100_000
    lr_name: Literal['high', 'low', 'step'] = 'high'
    lr_step_init: float = 1e-3
    lr_step_size: int = 10_000
    lr_step_gamma: float = 0.5

    wandb_user: str = 'kosiokarchev'
    wandb_project: str = 'cigars'
    wandb_mode: str = 'offline'

    version: int = 0
    thresh: float = 1e-4


parser = argparse.ArgumentParser('CIGaRS')

parser.add_argument('--norms', action=argparse.BooleanOptionalAction, default=True)

parser.add_argument('--fine-tune', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--train-locals', action=argparse.BooleanOptionalAction, default=False)

parser.add_argument('--max-steps', default=100_000, type=int)
parser.add_argument('--lr-name', default='high', choices=('high', 'low', 'step'))
parser.add_argument('--lr-step-init', default=1e-3, type=float)
parser.add_argument('--lr-step-size', default=10_000, type=int)
parser.add_argument('--lr-step-gamma', default=0.5, type=float)

parser.add_argument('--wandb-user', default='kosiokarchev')
parser.add_argument('--wandb-project', default='cigars')
parser.add_argument('--wandb-mode', default='offline')

parser.add_argument('--version', default=0)
parser.add_argument('--thresh', default=1e-4)

parser.add_argument('--variant', choices=('dindep',), default='dindep')
parser.add_argument('counts', type=int)
parser.add_argument('zoom_stage', type=int)

args: Args = parser.parse_args()
print(args)
