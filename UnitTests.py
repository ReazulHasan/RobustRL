import unittest
import Dirichlet_Uncertainty_set
import Gaussian_Uncertainty_Set
from craam import crobust
import Utils
import math
import numpy as np


class TestAmbiguitySetMethods(unittest.TestCase):

    def test_find_nominal_point_2(self):
        points = [[0,0],[4,0]]
        nominal = find_nominal_point(np.asarray(points))
        self.assertEqual([2.0,0.0], nominal)
        
    def test_find_nominal_point_4(self):
        points = [[0,0],[0,4],[4,0],[4,4]]
        nominal = find_nominal_point(np.asarray(points))
        self.assertEqual([2.0,2.0], nominal)

    def test_find_nominal_point_multiple(self):
        points = [[0,0],[0,4],[4,0],[4,4],[1,1],[3,1],[4,2]]
        nominal = find_nominal_point(np.asarray(points))
        self.assertEqual([2.0,2.0], nominal)
        
    def test_get_uset(self):
        confidence = 0.5
        points = np.asarray([[0,0],[0,4],[4,0],[4,4],[1,1],[3,1],[4,2],[1,2]])
        confidence_rank = math.ceil(len(points)*confidence)
        nominal = find_nominal_point(points)
        uset = get_uset(points,nominal,confidence_rank)
        self.assertEqual([[1,2],[1,1],[3,1],[4,2]], uset[0].tolist())

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAmbiguitySetMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)