from __future__ import absolute_import, division, print_function
import salty
import salty.visualization as vis
import unittest
from keras.layers import Dense, Dropout
from keras.models import Sequential


class visualization_library_tests(unittest.TestCase):
    data = ['cpt', 'density']
    T = [298.1, 298.16]
    P = [101, 102]
    devmodel = salty.aggregate_data(data, T=T, P=P)
    X_train, Y_train, X_test, Y_test = \
        salty.devmodel_to_array(devmodel, train_fraction=0.8)
    model = Sequential()
    model.add(Dense(75, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.25))
    model.add(Dense(Y_train.shape[1], activation='linear'))
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=['mse'])
    model.fit(X_train, Y_train, epochs=1, verbose=False)

    def test_1_parity_plot(self):
        plot = vis.parity_plot(self.X_test, self.Y_test, self.model,
                               self.devmodel)
        return plot

    def test_benchmark(self):
        salty.Benchmark.run(self.test_1_parity_plot)


if __name__ == '__main__':
    unittest.main()
