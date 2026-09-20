import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_pipeline import FeatureEncoder, prepare_split


class FeatureEncoderTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame(
            {
                "a": [0.0, 10.0, 5.0, 2.5, 7.5, 1.0, 9.0, 4.0, 6.0, 3.0],
                "b": [100.0, 200.0, 150.0, 125.0, 175.0, 110.0, 190.0, 140.0, 160.0, 130.0],
            }
        )
        self.labels = pd.Series(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])

    def test_feature_encoding_is_binary_and_eight_bits_per_feature(self):
        encoder = FeatureEncoder.fit(self.features, self.labels)
        encoded = encoder.transform_features(self.features)

        self.assertEqual(encoded.shape, (10, 16))
        self.assertTrue(np.isin(encoded, [0, 1]).all())

    def test_label_encoding_round_trips(self):
        encoder = FeatureEncoder.fit(self.features, self.labels)
        encoded = encoder.transform_labels(self.labels)
        decoded = encoder.decode_labels(encoded)

        np.testing.assert_array_equal(decoded, self.labels.to_numpy(dtype=object))

    def test_split_is_deterministic_and_stratified(self):
        first = prepare_split(self.features, self.labels, test_size=0.4, random_state=7)
        second = prepare_split(self.features, self.labels, test_size=0.4, random_state=7)

        np.testing.assert_array_equal(first.x_train, second.x_train)
        np.testing.assert_array_equal(first.y_test_raw, second.y_test_raw)
        self.assertEqual(set(first.y_test_raw), {"A", "B"})

    def test_test_values_are_clipped_to_training_range(self):
        train = pd.DataFrame({"a": [0.0, 10.0]})
        labels = pd.Series(["A", "B"])
        encoder = FeatureEncoder.fit(train, labels)
        encoded = encoder.transform_features(pd.DataFrame({"a": [-100.0, 100.0]}))

        np.testing.assert_array_equal(encoded[0], np.zeros(8, dtype=np.uint8))
        np.testing.assert_array_equal(encoded[1], np.ones(8, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
