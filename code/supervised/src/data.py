import glob
import sys
sys.path.append('../..')

import os
import logging
import numpy as np

from typing import List, Optional
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .args import DCNNConfig

logger = logging.getLogger(__name__)


class ImageCxDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images: Optional[List[torch.Tensor]] = None,  # batch, channel, height, width
                 lbs: Optional[List[int]] = None,
                 ):
        super().__init__()
        self._images = images
        self._lbs = lbs

    @property
    def n_insts(self):
        return len(self.images)

    @property
    def images(self):
        return self._images

    @property
    def lbs(self):
        return self._lbs

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._lbs is not None:
            return self._images[idx], self._lbs[idx]
        else:
            return self._images[idx]

    def load_file(self,
                  file_dir: str,
                  config: Optional[DCNNConfig] = None) -> "ImageCxDataset":
        """
        Load data from disk

        Parameters
        ----------
        file_dir: the directory of the file.
        config: chmm configuration.

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        images_0_paths = glob.glob(os.path.join(file_dir, 'NORMAL', '*.jpeg'))
        images_1_paths = glob.glob(os.path.join(file_dir, 'PNEUMONIA', '*.jpeg'))

        T = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((config.image_size, config.image_size)),
                                transforms.Normalize(0.5, 0.5)])

        image_0_list = [T(Image.open(img_path)) for img_path in images_0_paths]
        image_1_list = [T(Image.open(img_path)) for img_path in images_1_paths]

        self._images = image_0_list + image_1_list
        self._lbs = [0] * len(image_0_list) + [1] * len(image_1_list)

        return self

    def pop_random(self, ratio: Optional[float] = 0.15):

        rand_choice = np.random.binomial(1, ratio, len(self))
        imgs_keep = list()
        lbs_keep = list()
        imgs_output = list()
        lbs_output = list()
        for img, lb, choice in zip(self._images, self._lbs, rand_choice):
            if choice == 1:
                imgs_output.append(img)
                lbs_output.append(lb)
            elif choice == 0:
                imgs_keep.append(img)
                lbs_keep.append(lb)
            else:
                raise ValueError(f'Invalid choice: {choice}')
        self._images = imgs_keep
        self._lbs = lbs_keep

        return ImageCxDataset(imgs_output, lbs_output)
