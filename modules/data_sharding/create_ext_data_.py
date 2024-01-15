import random
from random import randrange
import numpy
import numpy as np
from data_sharding.helper import shuffle_pairwise
from tqdm import tqdm


class ExtData:

    def __init__(self,
                 private_shards,
                 private_features_combined,
                 private_labels_combined,
                 proxy_data_perc=10, # not used
                 ext_features=None,
                 ext_labels=None,
                 data_sharing=False,
                 random_labels=False,
                 num_classes=10):
        """

        :param private_shards (dictionary): keys: client name, values (tuple): list of features, list of labels
        :param private_features_combined (list): all client features combined
        :param private_labels_combined (list): all client labels combined
        :param ext_features (list): all extension features combined
        :param ext_labels (list): all extension labels combined
        :param data_sharing (boolean): if to use the data extension technique
        :param random_labels (boolean): if to order the labels or randomize them
        :param num_classes (int): number of classes in the task
        """
        self.private_shards = private_shards
        self.private_features_combined = private_features_combined
        self.private_labels_combined = private_labels_combined
        self.proxy_data_perc = proxy_data_perc # not used
        self.ext_features = ext_features
        self.ext_labels = ext_labels
        self.num_classes = num_classes
        self.random_labels = random_labels
        self.data_sharing = data_sharing
        self.classes = set(private_labels_combined)
        self.extension_shards = None

        # if no ext data was given and we are not using the data sharing method
        # create synthetic ext data using pseudo-patterns
        if ext_features is None and not self.data_sharing:
            print('[INFO] Creating pseudo-pattherns ...')
            self.ext_features, self.ext_labels = self.create_pseudo_patterns_ext_data()
            self.extension_shards = self.create_ext_data_for_all_client()
            print('[INFO] Done!')

        # we are using data sharing
        elif ext_features is None and self.data_sharing:
            print('[INFO] Creating central data to be shared ...')
            self.ext_features, self.ext_labels = self.create_data_sharing_ext_data()
            print('[INFO] Done!')
            self.extension_shards = self.create_ext_data_for_all_client()
        else:
            self.extension_shards = None

    def create_pseudo_patterns_ext_data(self):
        """
        Create pseudo patter based ext_features and assign labels for a single class
        :return: a tupple containg a list of features and a list of labels
        """
        # assuming a balance dataset: we use the number samples from a random shard
        # as the number of pseudo-patterns to be created for each class
        single_shard_features, single_shard_labels = next(iter(self.private_shards.values()))
        data_dimension = single_shard_features.shape[1:]  # shape of a single feature
        samples_per_class = len(single_shard_labels)
        labels = range(self.num_classes)
        ext_features, ext_labels = [], []
        for label in tqdm(labels):
            # features = [self.single_pseudo_pattern() for _ in range(samples_per_class)]
            features = [self.single_pseudo_pattern(dim=data_dimension) for _ in range(samples_per_class)]
            ext_features.extend(features)
            ext_labels.extend([label] * samples_per_class)

        return ext_features, ext_labels

    def create_data_sharing_ext_data(self):
        return self.private_features_combined, self.private_labels_combined

    def single_pseudo_pattern(self, dim=(32, 32, 3)):
        feature_sample = np.random.randint(255, size=dim)
        return feature_sample

    def create_ext_data_for_all_client(self):
        """
        Create the extension datasets for each client.
        """
        if float(self.proxy_data_perc) == 0.:
                return
        extension_shards = {}
        
        for key, value in tqdm(self.private_shards.items()):
            num_samples = len(value[1])
            missing_labels = [i for i in self.classes if i not in np.unique(value[1])]

            # only select extension samples belong to the missing classes in this primary shard
            ext_data = [(feat, lab) for (feat, lab) in zip(self.ext_features, self.ext_labels) if lab in missing_labels]
            random.shuffle(ext_data)
            num_proxy_samples = int(self.proxy_data_perc * num_samples)
            ext_data = ext_data[:num_proxy_samples]
            # proxy_data_quantity = int(num_samples * self.proxy_data_perc)
            #ext_data = ext_data[:proxy_data_quantity]
            ext_data_features, ext_data_labels = zip(*ext_data)

            if self.random_labels:
                random.shuffle(ext_data_labels)
            extension_shards[key] = (np.array(ext_data_features), np.array(ext_data_labels))
        return extension_shards

    def add_ext_single_client(self, client_id, fraction_of_ext_data):
        private_features, private_labels = self.private_shards[client_id]
        extension_features, extension_labels = self.extension_shards[client_id]

        # quantity of extension samples to use as % of the number of private shard samples
        idx = int(fraction_of_ext_data * len(private_labels))
        final_features = list(private_features) + list(extension_features)[:idx]
        final_labels = list(private_labels) + list(extension_labels)[:idx]
        final_features, final_labels = shuffle_pairwise(final_features, final_labels)
        return np.array(final_features), np.array(final_labels)
