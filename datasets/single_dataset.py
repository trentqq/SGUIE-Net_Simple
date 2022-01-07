from datasets.base_dataset import BaseDataset, get_transform, get_divide_masks, get_test_divide_masks
from util.util import make_dataset
from PIL import Image
import os
import torchvision.transforms as transforms


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        #self.img_size = opt.crop_size
        self.img_size = (opt.image_height, opt.image_width)
        self.dir_raw = os.path.join(opt.dataroot, opt.phase, 'raw')  # raw
        self.dir_mask = os.path.join(opt.dataroot, opt.phase,  'mask')  # masks
        self.raw_paths = sorted(make_dataset(self.dir_raw, opt.max_dataset_size))

        self.transform_raw = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains raw and A_paths
            raw(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        raw_path = self.raw_paths[index]
        raw_name = os.path.split(raw_path)[1].split('.')[0]
        raw_img = Image.open(raw_path).convert('RGB')

        raw = self.transform_raw(raw_img)
        #mask = get_divide_masks(self.img_size, self.dir_divide_C, raw_name, self.transform_C)
        mask = get_test_divide_masks(self.img_size, self.dir_mask, raw_name, self.transform_mask)
        return {'raw': raw, 'mask': mask, 'raw_paths': raw_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.raw_paths)
