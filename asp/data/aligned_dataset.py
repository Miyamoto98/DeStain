import os.path
import numpy as np
import torch
import json

from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import cv2


def brightness_normalize(img, target_mean=128.0):
    """Per-image brightness normalization in LAB space.
    img: PIL RGB image.
    Returns: PIL RGB image with L-channel mean shifted to target_mean.
    """
    arr = np.array(img, dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.float32)
    l = l - l.mean() + target_mean
    l = np.clip(l, 0, 255).astype(np.uint8)
    lab[:, :, 0] = l
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


class ReinhardNormalizer:
    def __init__(self, ref_path):
        data = np.load(ref_path)
        self.ref_mean = data["mean"].astype(np.float32)
        self.ref_std = data["std"].astype(np.float32)
        self.eps = 1e-6

    def __call__(self, img):
        arr = np.array(img, dtype=np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        for c in range(3):
            mean = lab[:, :, c].mean()
            std = lab[:, :, c].std()
            lab[:, :, c] = (lab[:, :, c] - mean) / (std + self.eps) * self.ref_std[c] + self.ref_mean[c]
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class AlignedDataset(BaseDataset):
    """
    This dataset class can load aligned/paired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        assert self.A_size == self.B_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index = random.randint(0, self.A_size - 1)
            index_B = index % self.B_size
            
        A_path = self.A_paths[index]  # make sure index is within then range
        B_path = self.B_paths[index_B]

        assert A_path == B_path.replace('trainB', 'trainA').replace('valB', 'valA').replace('testB', 'testA')

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.opt.brightness_norm:
            A_img = brightness_normalize(A_img)

        if self.opt.reinhard_norm:
            if not hasattr(self, '_reinhard'):
                self._reinhard = ReinhardNormalizer(self.opt.reinhard_ref)
            A_img = self._reinhard(A_img)

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)

        transform_params = get_params(modified_opt, A_img.size)
        if self.opt.bottom_right_crop:
            w, h = A_img.size
            x = max(0, w - modified_opt.crop_size)
            y = max(0, h - modified_opt.crop_size)
            transform_params['crop_pos'] = (x, y)
            transform_params['flip'] = False

        transform = get_transform(modified_opt, transform_params)

        # FDL: synchronize transforms
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        A = transform(A_img)
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)