import argparse
from os.path import join, exists
import numpy as np
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST, ImageNet
from utils import mkdirs


data_name_vs_pytorch_obj = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "mnist": MNIST,
    "mnist_fashion": FashionMNIST,
    "imagenet": ImageNet
}


def download_dataset(datasetname, download_dir):
    dataset_obj = data_name_vs_pytorch_obj[datasetname]
    train_dataset = dataset_obj(download_dir, train=True, download=True)
    test_dataset = dataset_obj(download_dir, train=False, download=True)
    return train_dataset, test_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='local dataset root dir')
    parser.add_argument('--datasetname', type=str, help='name of the dataset, e.g cifar10')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    download_dir = join(args.root_dir, args.datasetname)
    if not exists(download_dir):
        mkdirs(download_dir)
        train_dataset, test_dataset = download_dataset(args.datasetname, download_dir)

        # convert to npy arrays
        train_feature_array, train_label_array = train_dataset.data, np.array(train_dataset.targets)
        test_feature_array, test_label_array = test_dataset.data, np.array(test_dataset.targets)

        # save as npy arrays
        np.save(join(download_dir, './train_features.npy'), train_feature_array)
        np.save(join(download_dir, './train_labels.npy'), train_label_array)
        np.save(join(download_dir, './test_features.npy'), test_feature_array)
        np.save(join(download_dir, './test_labels.npy'), test_label_array)
    print(f"dataset ({args.datasetname}) is ready and available here: {download_dir}")


