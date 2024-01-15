import random
import numpy as np


def shuffle_pairwise(list1, list2):
    """"
    Takes in 2 list and shuffles them pairwise
    """
    zipped_list = list(zip(list1, list2))
    random.shuffle(zipped_list)
    shuffled_list1, shuffled_list2 = zip(*zipped_list)
    return list(shuffled_list1), list(shuffled_list2)


def create_subarrays(input_array, num_sub_arrays):
    """
    Creates 'num_sub_array' unmber of equal length subarrays from
    an input array.
    :param input_array: input array
    :param num_sub_arrays: number of sub-arrays to be returned
    :return:
    """
    sub_arrays = []
    for i in range(0, len(input_array), num_sub_arrays):
        sub_arrays.append(input_array[i: i + num_sub_arrays])
    return sub_arrays


def pseudo_pattern_ordered_labels(data_shards, num_classes):
    """
    Generate pseudo pattern samples for each class, these samples will be added
    to each private data shards based on the missing classes
    :param data_shards: dictionary of private data shards
    :param num_classes: number of class in the overall data
    :return: feature and labels in numpy array format
    """
    sample_features, sample_labels = list(data_shards.values())[0]
    total_num_pseudo = len(sample_labels) * len(data_shards)
    num_pseudo_per_class = int(total_num_pseudo / num_classes)
    classes = [i for i in range(num_classes)]
    pseudo_samples_by_class = []
    for lab in classes:
        feature = np.random.randint(255, size=sample_features[0].shape)
        pseudo_samples_by_class.extend([(feature, lab) for _ in range(num_pseudo_per_class)])
    random.shuffle(pseudo_samples_by_class)
    X, Y = zip(*pseudo_samples_by_class)
    print('[INFO] Pseudo pattern with shape{} with ordered labels will be used as the ext data'.format(
        np.array(X).shape))
    return np.array(X), np.array(Y)


def pseudo_pattern_random_labels(data_shards):
    """
    Generate pseudo pattern samples for each class but with randomly assigned labels
    from the classes in the overall data set. These samples will be added
    to each private data shards based on the missing classes
    Perform
    :param data_shards: private data shards
    :return: feature and labels in numpy array format
    """
    sample_features, sample_labels = list(data_shards.values())[0]
    total_num_pseudo = len(sample_labels) * len(data_shards)
    pseudo_samples_random_label = [(np.random.randint(255, size=sample_features[0].shape),
                                    np.random.randint(10)) for _ in range(total_num_pseudo)]
    features, labels = zip(*pseudo_samples_random_label)
    print('[INFO] Pseudo pattern with shape{} with random labels will be used as the ext data'.format(
        np.array(features).shape))
    return np.array(features), np.array(labels)


def sample_client(list_if_shard_ids: list, num_samples):
    return random.sample(list_if_shard_ids, num_samples)
