import numpy as np
import random
import sys
from data_sharding.helper import shuffle_pairwise, create_subarrays


class Shard:
    def __init__(self, features: list, labels: list, num_shards=10):
        self.features, self.labels = shuffle_pairwise(features, labels)
        self.num_shards = num_shards
        self.iid_shards = self.iid_sharding()

    @staticmethod
    def create_shards(feature_list, label_list, samples_per_shard, initial=''):
        """
        Partitions a the list of features and labels into a desire number of shards
        :param feature_list: list of features in the data set
        :param label_list: list of labels
        :param samples_per_shard: sample count in each of the shard to be created
        :param initial: for creating a unique key for each shard in the final hash table
        :return: an hastable whose values are tuples containing a list of features and a list
        of labels for each generated data shard.
        """
        shards = {}
        for i in range(0, len(label_list), samples_per_shard):
            feature_shard = feature_list[i: i + samples_per_shard]
            label_shard = label_list[i: i + samples_per_shard]

            # the remaining samples that are not up to a full shard should be discarded
            if len(label_shard) < samples_per_shard:
                continue
            shard_key = initial + str(i)
            shards[shard_key] = (np.array(feature_shard), np.array(label_shard))
        return shards

    def iid_sharding(self):
        """
        Creates "self.num_shards" number of iid sub-datasets from a
        list of features and a list labels
        """
        samples_per_shard = len(self.labels) // self.num_shards
        return self.create_shards(self.features, self.labels, samples_per_shard)

    def niid_sharding(self, num_classes_per_shard=1):
        """
        Create "num_shards" number of niid sub-dataset from a list
        of features and a list labels.
        :param num_classes_per_shard: number of label classes a single shard can hold
        :return shards of niid data
        """
        unique_labels = np.unique(np.array(self.labels))
        assert len(unique_labels) % num_classes_per_shard == 0, 'number of unique_labels must be divisible by ' \
                                                                'num_classes_per_shard '
        random.shuffle(unique_labels)
        sub_labels = create_subarrays(list(unique_labels), num_classes_per_shard)

        shards = {}
        for i, sub_label in enumerate(sub_labels):
            # collect all the data samples belonging to this sub-label list
            sub_samples = [(ft, lab) for (ft, lab) in zip(self.features, self.labels) if lab in sub_label]
            features, labels = zip(*sub_samples)
            # create shards out of each sub-samples, if for example we need to create 10 clients each holding
            # data samples belong to only 2 label classes, then each sub_sample
            # must be further splited into 2 shards
            initial = '{}_'.format(i)
            samples_per_shard = len(labels) * len(sub_labels) // self.num_shards
            sub_sample_shards = self.create_shards(features, labels, samples_per_shard, initial)
            shards.update(sub_sample_shards)
        return shards

    def get_iid_shards(self):
        return self.iid_shards

    def get_niid_shards(self, num_classes_per_shard):
        return self.niid_sharding(num_classes_per_shard)








