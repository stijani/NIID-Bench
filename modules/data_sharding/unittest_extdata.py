import numpy as np
import unittest
from create_ext_data import ExtData
from create_private_shards import Shard

main_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
main_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ext_feature = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
ext_labels = np.array([9, 1, 4, 7, 0, 2, 1, 5, 8, 3, 0, 7, 3, 5, 3, 6, 1, 6, 5, 2, 0, 8, 9, 2, 6, 9])


shard = Shard(main_features, main_labels, 2)


class ExtDataTest(unittest.TestCase):

    @staticmethod
    def test_dummy_data():
        # get niid shards
        niid_shards = shard.get_niid_shards(5)
        extdata = ExtData(niid_shards)
        print('\n', '******* private niid shards ********')
        for key, value in niid_shards.items():
            print(key, value)

        # add extension data to each shard
        ext_shards = extdata.add_ext(np.array(ext_feature), np.array(ext_labels), 0.3)
        print('\n', '******* extended niid shards ********')
        for key, value in ext_shards.items():
            print(key, value)

        # share a percentage of data within each shard with one another
        shared_shards = extdata.data_share(0.1)
        print('\n', '******* private shared plus data ********')
        for key, value in shared_shards.items():
            print(key, value)


if __name__ == '__main__':
    unittest.main()