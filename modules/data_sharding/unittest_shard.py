import unittest
from create_private_shards import Shard

features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

shard = Shard(features, labels, 10)


class ShardTest(unittest.TestCase):
    @staticmethod
    def test_dummy_data():
        # get iid shards
        iid_shards = shard.get_iid_shards()
        print('******* iid shards ********')
        for key, value in iid_shards.items():
            print(key, value)

        # get niid shards
        niid_shards = shard.get_niid_shards(2)
        print('\n', '******* niid shards ********')
        for key, value in niid_shards.items():
            print(key, value)


if __name__ == '__main__':
    unittest.main()