import random
from random import randrange
import numpy
import numpy as np
from data_sharding.helper import shuffle_pairwise
import sys


class ExtData:

    def __init__(self, private_data: dict, num_classes: int):
        self.private_data = private_data
        self.num_classes = num_classes
        

    def add_ext(self, ext_features: numpy.ndarray, ext_labels: numpy.ndarray, frac, random_labels=False):
        """
        Extends each client private data with a secondary data set.
        :param ext_features: features of extension data
        :param ext_labels: labels of the enxtension data set
        :param frac: percentage of the ext data samples to be used relative to the primary samples
        :param random_labels: if to randomise the ext class labels or keep them in order
        :param num_classes: number class in the primary task, e.g for cifiar10 it is 10
        :return: a dictionary with individual data shards as its values
        """
        private_plus_ext_shards = {}
        unique_ext_labels = np.unique(ext_labels)
        for key, value in self.private_data.items():
            missing_labels = [i for i in unique_ext_labels if i not in np.unique(value[1])]
            ext_data = [(feat, lab) for (feat, lab) in zip(ext_features, ext_labels) if lab in missing_labels]
            ext_data_features, ext_data_labels = zip(*ext_data)
            ext_data_features, ext_data_labels = shuffle_pairwise(ext_data_features, ext_data_labels)

            if random_labels:
                random.shuffle(ext_data_labels)
            idx = int(frac * len(value[1]))
            final_features = list(value[0]) + list(ext_data_features)[:idx]
            final_labels = list(value[1]) + list(ext_data_labels)[:idx]
            final_features, final_labels = shuffle_pairwise(final_features, final_labels)
            private_plus_ext_shards[key] = (np.array(final_features), np.array(final_labels))
        return private_plus_ext_shards

    def add_ext_single_client(self, client_id, ext_features, ext_labels, frac, random_labels=False):
        """
        Extends each client private data with a secondary data set.
        :param ext_features: features of extension data
        :param ext_labels: labels of the enxtension data set
        :param frac: percentage of the ext data samples to be used relative to the primary samples
        :param random_labels: if to randomise the ext class labels or keep them in order
        :param client_id: id of the client we are adding ext. data to
        :return: tuple of numpy array representing  the extended features and targets
        """
        unique_ext_labels = np.unique(ext_labels)
        features, targets = self.private_data[client_id]
        #missing_labels = [i for i in unique_ext_labels if i not in np.unique(targets)]
        missing_labels = [i for i in unique_ext_labels if i not in np.unique(targets) and i < self.num_classes]
        ext_data = [(feat, lab) for (feat, lab) in zip(ext_features, ext_labels) if lab in missing_labels]
        ext_data_features, ext_data_targets = zip(*ext_data)
        ext_data_features, ext_data_targets = shuffle_pairwise(ext_data_features, ext_data_targets)

        if random_labels:
            random.shuffle(ext_data_targets)
        idx = int(frac * len(ext_data_targets))
        final_features = list(features) + list(ext_data_features)[:idx]
        final_labels = list(targets) + list(ext_data_targets)[:idx]
        final_features, final_labels = shuffle_pairwise(final_features, final_labels)
        return np.array(final_features), np.array(final_labels)


    def data_share(self, frac):
        """
        Pick a fraction of data from each private shards and add the conbine lot back to each shard
        :param frac: faction or percention of the shared quantity relation to the qqty in the private shards
        :return: a dictionary with individual data shards as its values
        """
        private_plus_shared_shards = {}
        shared_features, shared_labels = [], []
        for key, value in self.private_data.items():
            shared_split = int(len(value[1]) * frac)
            shared_features.extend(value[0][:shared_split])
            shared_labels.extend(value[1][:shared_split])
        shared_features, shared_labels = shuffle_pairwise(shared_features, shared_labels)
        shared_features, shared_labels = np.array(shared_features), np.array(shared_labels)
        # add the shared data portion to the private data of each client
        for key, value in self.private_data.items():
            assert type(value[0]) == type(shared_features), 'the shared and private data must have the same type'
            assert type(value[1]) == type(shared_labels), 'the shared and private data must have the same type'
            features = list(np.concatenate((value[0], shared_features)))
            labels = list(np.concatenate((value[1], shared_labels)))
            features, labels = shuffle_pairwise(features, labels)
            private_plus_shared_shards[key] = (np.array(features), np.array(labels))
        return private_plus_shared_shards










