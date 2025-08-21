from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from ruamel.yaml import YAML

from clipppy.contrib.sbi_helper import SBIHelper, SavedProperty
from clipppy.utils.messengers import MultiContext


@dataclass
class CigarsHelper(SBIHelper):
    @property
    def cosmogroup(self):
        return ('Om0', 'w0', 'wa') if 'w0wa' in self.variant else ('Om0', 'Ode0')
    dtdgroup = 'dtd_A', 'dtd_s'

    variant: str
    counts: int

    @classmethod
    def from_args(cls):
        from .cli import args
        return cls(args.variant, args.counts, zoom_stage=args.zoom_stage)

    paper = Path('paper')

    base_traindir = Path('train/cigars')

    labels: ClassVar = {
        'Om0': r'$\Omega_{\mathrm{m}0}$', 'Ode0': r'$\Omega_{\Lambda 0}$',
        'dtd_A': r'$\log_{10} A$', 'dtd_s': '$s$',
        'step_M_logmass': r'$\Delta M$', 'step_M_logmass_loc': r'$\mathcal{M}_{\rm step}$',
        'gamma_M_logzsol': r'$\gamma_{[Z]}$', 'gamma_M_age': r'$\gamma_{\rm age}$',
        'M0': r'$M_0$', 'sigma_res': r'$\sigma_0$',
        'alpha': r'$\alpha$', 'beta': r'$\beta$', 'alpha_c': r'$\alpha_c$',

        'z': 'z', 'x_int': r'x_{\rm int}', 'c_int': r'c_{\rm int}', 'delta_M': r'\delta M',
        'dust_index': r'\delta', 'dust2': r'\tau_2',
        'logzsol': '[Z]', 'logmass': r'\mathcal{M}'
    }

    def __post_init__(self):
        self.basename = f'cigars-{self.variant}-{self.counts}'
        super().__post_init__()

    @property
    def bounds(self):
        return YAML(typ='safe').load(self.base_resdir / 'bounds.yaml')[self.basename][self.zoom_stage]

    if TYPE_CHECKING:
        cp_name: ClassVar[Path]

    cp: ClassVar[MultiContext] = SavedProperty(lambda self: self.traindir / 'cp.pt')
    norms = SavedProperty(lambda self: self.traindir / 'norms.pt')

    @property
    def model(self, **kwargs):
        from libcigars.cigars.model import CIGARS

        if 'dust_model' not in kwargs:
            if 'dindep' in self.variant:
                kwargs['dust_model'] = CIGARS.DustModel.independent
            elif 'dglob' in self.variant:
                kwargs['dust_model'] = CIGARS.DustModel.match_global
            elif 'dloc' in self.variant:
                kwargs['dust_model'] = CIGARS.DustModel.match_local

        if 'cosmodel' not in kwargs:
            if 'w0wa' in self.variant:
                from libcigars.cigars.cosmo import Flatw0waCDMCosmo
                kwargs['cosmodel'] = Flatw0waCDMCosmo()

        return CIGARS(self.counts, **kwargs)

    def datamod(self, sbi, batch_size, **kwargs):
        from libcigars.data import CigarsDataModule

        return CigarsDataModule(sbi, batch_size, traindir=self.traindir, labels=self.labels, **kwargs)
