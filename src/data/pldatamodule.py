import logging
import numpy as np
import h5py
import os
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import open_dict
from scipy.interpolate import interp1d

from .normalization import Nyul
from . import thdataset


class MRIDataModule(pl.LightningDataModule):
    """Data Module for Smartcontrast

    DO NOT move the data loader construction. It does not work in prepare or in setup,
    because h5py cannot be pickled and hence transfered.
    Number of workers depends on the strategy:
    --> For dm (multi gpu training on 1 machine) higher num_workers works.
    --> For dm_spawn (multi gpu training on multi machine) only 0 num_workers works.
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.hdf5_path = config.paths.hdf5
        if os.path.isdir(self.hdf5_path):
            self.hdf5_path = os.path.join(self.hdf5_path,
                                          sorted([f for f in os.listdir(self.hdf5_path) if f.endswith('.hdf5')],
                                                 key=lambda x: os.path.getmtime(os.path.join(self.hdf5_path, x)))[-1])
            logging.info('Using hdf5 file:' + self.hdf5_path)
        if self.cfg.data.nyul_normalization:
            database = h5py.File(self.hdf5_path, 'r')
            with open_dict(self.cfg):
                standard_scale = np.zeros(Nyul.NUM_LANDMARKS, dtype=np.float32)
                dataset = pd.read_csv(self.cfg.paths.train_csv)
                for key in dataset.index:
                    p_key = dataset.loc[key, 'patient']
                    landmarks = database[p_key]['T1_zero'].attrs['landmarks']
                    # The final region of interest should be between [0, 1].
                    f = interp1d(landmarks[[0, -1]], np.array([0., 1.], dtype=np.float32))
                    standard_scale += f(landmarks)

                self.standard_scale = standard_scale / len(dataset)
                # Omegaconf only supports string lists.
                self.cfg.data["standard_scale"] = [str(key) for key in self.standard_scale]
            database.close()

    def train_dataloader(self):
        patch_size = self.cfg.train.patch_size
        train_dataset = thdataset.MRIDataset(
            self.cfg.paths.train_csv, self.hdf5_path,
            multiplier=self.cfg.train.batch_size,
            cfg=self.cfg.data,
            patch_size=(patch_size, patch_size, patch_size),
            mm_location_csv=self.cfg.paths.mm_location_csv,
            mm_probability=self.cfg.train.mm_probability,
            use_weighted_sampling=self.cfg.train.use_weighted_sampling,
            training=True,
            transform=transforms.Compose([
                thdataset.DiederAugmentation(),
                thdataset.ToTensor(),
                ])
            )
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=self.cfg.train.batch_size,
                                  shuffle=True, drop_last=False, persistent_workers=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = thdataset.MRIDataset(
            self.cfg.paths.val_csv, self.hdf5_path,
            cfg=self.cfg.data,
            crop_method="center",
            patch_size=(256, 256, 256),
            transform=transforms.Compose([thdataset.ToTensor()])
        )
        val_loader = DataLoader(val_dataset, num_workers=4, batch_size=1,
                                shuffle=False, drop_last=False, persistent_workers=True)
        return val_loader
