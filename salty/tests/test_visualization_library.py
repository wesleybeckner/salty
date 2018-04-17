from __future__ import absolute_import, division, print_function
import salty
import salty.visualization as vis
import unittest
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential

class visualization_library_tests(unittest.TestCase):
    data = ['cpt', 'density']
    T = [298.1, 298.16]
    P = [101, 102]
    devmodel = salty.aggregate_data(data, T=T, P=P)

    def test_1_parity_plot(self):
        devmodel = salty.aggregate_data(self.data, T=self.T, P=self.P)
        X_train, Y_train, X_test, Y_test =\
            salty.devmodel_to_array(devmodel, train_fraction=0.8)
        model = Sequential()
        model.add(Dense(75, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dropout(0.25))
        model.add(Dense(Y_train.shape[1], activation='linear'))
        model.compile(optimizer="adam",
                      loss="mean_squared_error",
                      metrics=['mse'])
        model.fit(X_train, Y_train, epochs=100, verbose=False)
        vis.parity_plot(X_test, Y_test, model, devmodel)
        return devmodel

    def test_benchmark(self):
        salty.Benchmark.run(self.test_1_parity_plot)


if __name__ == '__main__':
    unittest.main()
