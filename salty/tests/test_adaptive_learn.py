from __future__ import absolute_import, division, print_function
import salty
import salty.adaptive_learn as adl
import unittest


class adaptive_learn_tests(unittest.TestCase):
    data = ['cpt', 'density', 'viscosity']
    T = [298.1, 298.16]
    P = [101, 102]
    devmodel = salty.aggregate_data(data, T=T, P=P)

    def test_1_expand_convex_hull(self):
        targets = adl.expand_convex_hull(self.devmodel)
        return targets

    def test_benchmark(self):
        salty.Benchmark.run(self.test_1_expand_convex_hull)


if __name__ == '__main__':
    unittest.main()
