##########################################################
# CUSTOM DATASET CLASS
##########################################################
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose
from PIL import Image

torch.backends.cudnn.benchmark = True

# tranforms - cifar10
def transform_cifar10():
    transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4), # Data augmentation
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    return transform


#TRANSFORMS = {'cifar10': None, 'mnist': None}
TRANSFORMS = {'cifar10': transform_cifar10(),
              'cifar100': transform_cifar10(),
              'mnist': None,
              'breast_cancer': None
              }


class CustomImageDataset(Dataset):
    """
    A class that packs a data set together
    with its transforms and convert it to a
    torch tensor.

    data:
        features [np.arrays]
        labels [np.arrays]
        transforms [python dictionary] - contains
        all augmentation and transformation
    """

    def __init__(self, dataset, name='cifar10', use_aug=True):
        features, labels = dataset 
        self.name = name 
        self.use_aug = use_aug
        assert features.shape[0] == labels.shape[0]
        self.features = features
        self.labels = labels
        self.transforms = TRANSFORMS[name]

    def __getitem__(self, index):
        img, label = self.features[index], self.labels[index]
         # convert the image to channel first if needed
        if img.shape[0] > 3:
            img = np.transpose(img, (2, 0, 1))
        # transform the image or just normalize
        if self.use_aug and self.transforms:
            # first convert image from numpy array to a pil image
            img = self.numpy_2_pil_image(img) 
            img = self.transforms(img)
        else:
            # no transform, just normalize
            img = torch.Tensor((img / 255.))
            #label = torch.Tensor(label).long()
        return img, label

    def __len__(self):
        return self.features.shape[0]

    def numpy_2_pil_image(self, numpy_img):
        # convert to channel first if array is channel last
        if numpy_img.shape[-1] != 3:
            # pil only deals with channel last format 
            numpy_img = np.transpose(numpy_img, (1, 2, 0))
        img_pil = Image.fromarray(numpy_img.astype(np.uint8))
        return img_pil


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)