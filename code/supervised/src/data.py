import os
import glob
import numpy as np

from typing import List, Optional, Union
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .args import DCNNConfig


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
        item = {'imgs': torch.tensor(self._images[idx])}
        if self._lbs:
            item['lbs'] = torch.tensor(self._lbs[idx])
        return item

    def load_file(self,
                  file_dir: str,
                  config: DCNNConfig) -> "ImageCxDataset":
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

        if config.debug_mode:
            self._images = self._images[:50]
            self._lbs = self._lbs[:50]

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

    def select(self, ids: Union[List[int], np.ndarray, torch.Tensor]):
        """
        Select a subset of dataset

        Parameters
        ----------
        ids: instance indices to select

        Returns
        -------
        A BertClassificationDataset consists of selected items
        """
        if np.max(ids) >= self.n_insts:
            raise ValueError('Invalid indices: exceeding the dataset size!')
        images_ = [self._images[idx] for idx in ids]
        lbs_ = [self._lbs[idx] for idx in ids] if self._lbs else None
        return ImageCxDataset(images_, lbs_)
