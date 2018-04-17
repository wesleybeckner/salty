from __future__ import absolute_import, division, print_function
import salty
import unittest
import datetime


class data_manipulation_tests(unittest.TestCase):
    data = ['cpt', 'density']
    data_ranges = [[200, 1000], [800, 1400]]
    T = [298.1, 298.16]
    P = [101, 102]
    string_model_name = "cpt_density_viscosity"
    string_devmodel = salty.aggregate_data(string_model_name, T=T, P=P,
                                           data_ranges=data_ranges,
                                           impute=True)
    devmodel = salty.aggregate_data(data, T=T, P=P, data_ranges=data_ranges)

    def test_1_aggregate_data(self):
        devmodel = salty.aggregate_data(self.data, T=self.T, P=self.P)
        return devmodel

    def test_2_devmodel_to_array(self):
        X_train, Y_train, X_test, Y_test = salty.devmodel_to_array(
            self.devmodel, train_fraction=0.8)
        return X_train, Y_train, X_test, Y_test

    def test_3_merge_duplicates(self):
        vals, stds, size, salts = salty.merge_duplicates(self.devmodel)
        return vals, stds, size, salts

    def test_benchmark(self):
        salty.Benchmark.run(self.test_1_aggregate_data)
        salty.Benchmark.run(self.test_2_devmodel_to_array)
        salty.Benchmark.run(self.test_3_merge_duplicates)

    def check_data(self, df):
        startTime = datetime.datetime.now()

        def fnDisplay(message):
            display(message, startTime)


def display(message, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{}\t{}".format(timeDiff, message))


if __name__ == '__main__':
    unittest.main()
