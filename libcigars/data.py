from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import attr

from phytorchx.dataframe import TensorDataFrame, IndexableList
from phytorchx.vltensor import VLTensor
from clipppy.commands.lightning.data import AbstractDataFrameParameterSBIDataModule
from clipppy.contrib.distributed import DistributedDataset
from clipppy.sbi.persistent.netcdf_data import NetCDFDataFrame


@attr.s(eq=False, auto_attribs=True)
class CigarsDataModule(AbstractDataFrameParameterSBIDataModule):
    traindir: Path
    preload: bool = True
    ddp: bool = False
    ddp_val: bool = False

    if TYPE_CHECKING:
        from phytorchx.dataframe import AbstractTensorDataFrame

        train_dataset: Optional[AbstractTensorDataFrame] = None
        val_dataset: Optional[AbstractTensorDataFrame] = None

    obs_event_dims = dict(gmags_obs=1)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        self.train_name = self.traindir / f'train.nc'
        self.val_name = self.traindir / f'val.nc'

    def _preload(self, ds):
        return TensorDataFrame({key: IndexableList(
            [v.as_subclass(VLTensor) for v in val]
            if self.ddp else val
        ) if isinstance(val, list) else val for key in ds.keys for val in [ds[key]]})

    def _load_ds(self, name):
        ds = NetCDFDataFrame(name, keys=tuple(self.keys))
        return self._preload(ds) if self.preload else ds

    def _dataset(self, dataset, shuffle: bool):
        if self.ddp or (not shuffle and self.ddp_val):
            return DistributedDataset(dataset, shuffle=shuffle, batch=self.batch_size)
        else:
            return super()._dataset(dataset, shuffle)

    @cached_property
    def train_dataset(self):
        return self._load_ds(self.train_name)

    @cached_property
    def val_dataset(self):
        return self._load_ds(self.val_name)
