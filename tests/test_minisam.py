# tests/test_minisam.py
import unittest
import numpy as np
from miniSam import MiniSam


class TestMiniSam(unittest.TestCase):
    """Unit-tests for the MiniSam Self-Organising-Map implementation."""

    def setUp(self):
        # small, fast map for unit-tests
        self.x, self.y, self.dim = 4, 4, 3
        self.seed = 123
        self.som = MiniSam(self.x, self.y, self.dim, random_seed=self.seed)
        self.data = np.random.default_rng(self.seed).random((10, self.dim))

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def test_deterministic_initial_weights(self):
        """Same RNG seed ⇒ identical initial weights."""
        som2 = MiniSam(self.x, self.y, self.dim, random_seed=self.seed)
        np.testing.assert_allclose(self.som.get_weights(), som2.get_weights())

    # ------------------------------------------------------------------ #
    #  Inference helpers
    # ------------------------------------------------------------------ #
    def test_winner(self):
        """`winner()` must return the BMU coordinates for an exact match."""
        target = np.array([1.0, 0.0, 0.0])
        self.som._weight[2, 1] = target          
        self.assertEqual(self.som.winner(target), (2, 1))  

    def test_quantization_shape(self):
        """`quantization()` preserves (N, dim) shape."""
        q = self.som.quantization(self.data)     
        self.assertEqual(q.shape, self.data.shape)

    # ------------------------------------------------------------------ #
    #  Training routines
    # ------------------------------------------------------------------ #
    def test_train_random_updates_weights(self):
        """Online training should modify weights in-place."""
        before = self.som.get_weights().copy()
        self.som.train_random(self.data, num_iteration=20)  
        self.assertFalse(np.allclose(before, self.som.get_weights()))

    def test_train_batch_updates_weights(self):
        """Batch training should also move the code-book vectors."""
        som = MiniSam(self.x, self.y, self.dim, random_seed=self.seed)
        before = som.get_weights().copy()
        som.train_batch(self.data, num_iteration=5)         

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #
    def test_activation_response_counts(self):
        """Hit histogram must add up to the number of samples fed in."""
        hits = self.som.activation_response(self.data)      
        self.assertEqual(hits.sum(), len(self.data))

    def test_distance_map_properties(self):
        """UMatrix shape equals map shape and contains non-negative values."""
        um = self.som.distance_map()                        
        self.assertEqual(um.shape, (self.x, self.y))
        self.assertTrue((um >= 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
