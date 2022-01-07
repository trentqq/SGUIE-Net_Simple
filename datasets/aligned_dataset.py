import os.path
from datasets.base_dataset import BaseDataset, get_transform, get_params, get_divide_masks
from util.util import make_dataset
from PIL import Image
import random
import torch


class AlignedDataset(BaseDataset):
    """
    This dataset class can load aligned/paired datasets.

    It requires three directories to host training images
     - '/path/to/data/raw'  raw underwater images
     - '/path/to/data/ref'  ref enhancement images
     - '/path/to/data/mask'  GT semantic segmentation

    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare three directories:
    '/path/to/data/raw', '/path/to/data/ref' and '/path/to/data/mask' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.img_size = opt.crop_size
        self.dir_raw = os.path.join(opt.dataroot, opt.phase, 'raw')
        self.dir_ref = os.path.join(opt.dataroot, opt.phase, 'ref')
        self.dir_mask = os.path.join(opt.dataroot, opt.phase, 'mask')

        self.raw_paths = sorted(make_dataset(self.dir_raw, opt.max_dataset_size))  # load images from '/path/to/data/raw'

        self.raw_size = len(self.raw_paths)  # get the size of dataset A

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains raw, ref, mask, A_paths, B_paths and C_paths
            raw (tensor)       -- an image in the input domain
            ref (tensor)       -- its corresponding image in the target domain
            mask (tensor)       -- its GT semantic segmentation
            raw_paths (str)    -- image paths
            ref_paths (str)    -- image paths
            mask_paths (str)    -- image paths
        """
        raw_path = self.raw_paths[index % self.raw_size]  # make sure index is within then range


        raw_name = os.path.split(raw_path)[1]
        base_name = raw_name.split('.')[0]
        ref_path = os.path.join(self.dir_ref, raw_name)
        mask_path = os.path.join(self.dir_mask, base_name + '.bmp')


        raw_img = Image.open(raw_path).convert('RGB')
        ref_img = Image.open(ref_path).convert('RGB')

        # apply image transformation
        transform_params = get_params(self.opt, raw_img.size)
        self.transform_raw = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        self.transform_ref = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        self.transform_mask = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), norm=False)

        raw = self.transform_raw(raw_img)
        ref = self.transform_ref(ref_img)
        mask = get_divide_masks(self.img_size, self.dir_mask, base_name, self.transform_mask)

        return {'raw': raw, 'ref': ref, 'mask': mask, 'raw_paths': raw_path, 'ref_paths': ref_path, 'mask_paths': mask_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.raw_size
