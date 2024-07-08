import ast
import h5py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Union

import logging

from . import SeriesData


class Dieder2DAugmentation(object):
    """Augment the data by tranformation in Dieder Group.
    """
    def _identity(self, x):
        return x

    def _rflip(self, x):
        return x[:, ::-1, :]

    def _lflip(self, x):
        return x[:, :, ::-1]

    def _lrflip(self, x):
        return x[:, ::-1, ::-1]

    def __init__(self, rotate=True, flip=True):
        self._rotate = rotate
        self._flip = flip

        if self._flip:
            self._flip_ops = [self._identity,
                              self._lflip,
                              self._rflip,
                              self._lrflip]

    def __call__(self, sample):
        if self._rotate:
            choice = np.random.choice(4)
            for k in sample.keys():
                if isinstance(sample[k], np.ndarray):
                    sample[k] = np.ascontiguousarray(np.rot90(sample[k], choice, axes=(1, 2)))

        if self._flip:
            choice = np.random.choice(self._flip_ops)
            for k in sample.keys():
                if isinstance(sample[k], np.ndarray):
                    sample[k] = np.ascontiguousarray(choice(sample[k]))

        return sample


class DiederAugmentation(object):
    """Augment the data by tranformation in Dieder Group.
    """
    def _identity(self, x):
        return x

    def _lflip(self, x):
        return x[:, ::-1, :, :]

    def _rflip(self, x):
        return x[:, :, ::-1, :]

    def _dflip(self, x):
        return x[::-1, :, :, :]

    def _lrflip(self, x):
        return x[:, ::-1, ::-1, :]

    def _dlflip(self, x):
        return x[::-1, ::-1, :, :]

    def _drflip(self, x):
        return x[::-1, :, ::-1, :]

    def _dlrflip(self, x):
        return x[::-1, ::-1, ::-1, :]

    def __init__(self, rotate=True, flip=True, transpose=True):
        self._rotate = rotate
        self._flip = flip
        self._transpose = transpose

        if self._flip:
            self._flip_ops = [self._identity,
                              self._lflip,
                              self._rflip,
                              self._dflip,
                              self._lrflip,
                              self._dlflip,
                              self._drflip,
                              self._dlrflip]

    def __call__(self, sample):

        if self._rotate:
            choice = np.random.choice(4)
            for k in sample.keys():
                if isinstance(sample[k], np.ndarray) and sample[k].ndim == 4:
                    sample[k] = np.ascontiguousarray(np.rot90(sample[k], choice, axes=(1, 2)))

        if self._flip:
            choice = np.random.choice(self._flip_ops)
            for k in sample.keys():
                if isinstance(sample[k], np.ndarray) and sample[k].ndim == 4:
                    sample[k] = np.ascontiguousarray(choice(sample[k]))

        if self._transpose:
            permutation = tuple(np.random.permutation(3)) + (3,)
            for k in sample.keys():
                if isinstance(sample[k], np.ndarray) and sample[k].ndim == 4:
                    sample[k] = np.transpose(sample[k], permutation)

        return sample


class ToTensor(object):
    """Convert sample ndarrays to tensors.
    """

    @staticmethod
    def to_torch(np_array):
        if np_array.ndim == 4:
            return torch.from_numpy(np.transpose(np_array, (3, 0, 1, 2)))
        else:
            return torch.from_numpy(np_array)

    def __call__(self, sample):
        th_sample = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                th_sample[k] = ToTensor.to_torch(v)
            elif isinstance(v, int) or isinstance(v, float):
                th_sample[k] = v
            else:
                raise RuntimeError("Only np.ndarray and list objects are supported to convert to torch!")

        return th_sample


class MRIDataset(Dataset):
    """Image denoising data set."""

    def __init__(self, csv_file: str, database_file: str,
                 cfg: dict,
                 patch_size: Union[tuple, int] = None,
                 crop_method: str = "random",
                 mm_location_csv: str = None,
                 mm_probability: float = 0,
                 use_weighted_sampling: bool = False,
                 transform: list = None,
                 multiplier: int = 1,
                 training: bool = False):
        """
        Args:
            csv_file: Path to the csv data set descirption file.
            database_file: Path to the hdf5 file holding the database.
            cfg: Configuration dictionary of dataset using the following keys:
            patch_size: Size of extracted patch (x,y,z).
            crop_method: Either 'random' for random patch selection or 'center'
            mm_location_csv: Path to csv file that holds the locations of
                detected micro metastasis
            mm_probability: The probability of sampling a micro metastasis patch
            transform: Optional transform to be applied
                on a sample.
            multiplier: Prolonges the dataset by including copies of itself,
                without requiring extra memory.
            training: Specify if this dataset is used for training or validation
        """

        self.dataset = pd.read_csv(csv_file)
        self.database = h5py.File(database_file)

        self.training = training
        self.cfg = cfg
        self.weighted_sampling = use_weighted_sampling
        if self.weighted_sampling:
            self.probability_list = self.dataset.sample_percent.to_numpy() / self.dataset.sample_percent.to_numpy().sum()

        if patch_size:
            assert isinstance(patch_size, tuple)
            assert len(patch_size) == 3
        self.patch_size = patch_size
        assert crop_method in ["random", "center"]
        if crop_method == "random":
            # If the only possible value to extract a patch is 0 for randint, then we need the min.
            self.crop_method = lambda ps, shape: (np.random.randint(0, max(shape - ps, 1)) // 48) * 48
        else:
            self.crop_method = lambda ps, shape: max((shape - ps) // 2, 0)

        self.mm_probability = mm_probability
        if mm_location_csv:
            self.mm_locations = pd.read_csv(mm_location_csv, encoding='utf-8').loc[:, ["Kennung", "Metastasen"]]
            self.mm_locations = self.mm_locations[self.mm_locations["Metastasen"].str.len() > 2]

        self.transform = transform
        self.multiplier = multiplier

    def _sample_location(self, idx: int):
        if np.random.uniform() < self.mm_probability:
            # get a random case
            sample = self.mm_locations.sample()
            key = sample.iloc[0, 0]
            group = self.database[key]
            locations = ast.literal_eval(sample.iloc[0, 1])
            # get a random location
            z, y, x = locations[np.random.choice(len(locations))]
            # convert from center location to upper right corner
            assert self.patch_size  # we require a patch size in this case
            shape = group[SeriesData.zero].shape
            z = max(0, min(z - self.patch_size[0]//2, shape[0] - self.patch_size[0]))
            y = max(0, min(y - self.patch_size[1]//2, shape[1] - self.patch_size[1]))
            x = max(0, min(x - self.patch_size[2]//2, shape[2] - self.patch_size[2]))
            origin = (z, y, x)
        else:
            # regular sample
            if self.weighted_sampling:
                key = np.random.choice(self.dataset.patient.to_numpy(), 1,
                                       p=self.probability_list)[0]
            else:
                key = self.dataset.iloc[idx]["patient"]

            group = self.database[key]
            # define the patch selection
            origin = self._get_patch_origin(group[SeriesData.zero].shape)

        return key, group, origin

    def _get_patch_origin(self, shape):
        if self.patch_size:
            z = self.crop_method(self.patch_size[0], shape[0])
            y = self.crop_method(self.patch_size[1], shape[1])
            x = self.crop_method(self.patch_size[2], shape[2])
            return (z, y, x)
        else:
            return (0, 0, 0)

    def _get_patch(self, arr, origin, downsampled=False):
        if self.patch_size:
            if downsampled:
                z, y, x = origin
                z = z // 2
                y = y // 2
                x = x // 2
                return arr[z:z+self.patch_size[0] // 2,
                           y:y+self.patch_size[1] // 2,
                           x:x+self.patch_size[2] // 2]
            else:
                z, y, x = origin
                return arr[z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
        else:
            return arr[:]

    def __len__(self):
        return len(self.dataset) * self.multiplier

    def __getitem__(self, idx):
        # load the target image
        idx = idx % len(self.dataset)

        # define the patch selection
        key, group, origin = self._sample_location(idx)

        # convert to model input
        sd = SeriesData().load_h5(
            group, target=True,
            selector=lambda x: self._get_patch(x, origin),
        )
        # parse series data
        sample = sd.to_input(self.cfg, self.training)

        for k, v in sample.items():
            if not np.all(np.isfinite(v)):
                raise RuntimeError(f"NaN values detected in {key} {k}!")

        if self.transform:
            sample = self.transform(sample)

        return sample
