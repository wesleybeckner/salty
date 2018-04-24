from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


def parity_plot(X, Y, model, devmodel, axes_labels=None):
    """
    A standard method of creating parity plots between predicted and
    experimental values for trained models.

    Parameters
    ----------
    X: array
        experimental input data
    Y: array
        experimental output data
    model: model object
        either sklearn or keras ML model
    devmodel: dev_model object
        salty dev_model
    axes_labels: dict
        optional. Default behavior is to use the labels in the dev_model
        object.

    Returns
    ------------------
    plt: matplotlib object
        parity plot of predicted vs experimental values
    """
    model_outputs = Y.shape[1]
    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure(figsize=(2.5 * model_outputs, 2.5), dpi=300)
        for i in range(model_outputs):
            ax = fig.add_subplot(1, model_outputs, i + 1)
            minval = np.min([np.exp(model.predict(X)[:, i]), np.exp(Y)[:, i]])
            maxval = np.max([np.exp(model.predict(X)[:, i]), np.exp(Y)[:, i]])
            buffer = (maxval - minval) / 100 * 2
            minval = minval - buffer
            maxval = maxval + buffer
            ax.plot([minval, maxval], [minval, maxval], linestyle="-",
                    label=None, c="black", linewidth=1)
            ax.plot(np.exp(Y)[:, i], np.exp(model.predict(X))[:, i],
                    marker="*", linestyle="", alpha=0.4)
            if axes_labels:
                ax.set_ylabel("Predicted {}".format(
                              axes_labels['{}'.format(i)]))
                ax.set_xlabel("Actual {}".format(
                              axes_labels['{}'.format(i)]))
            else:
                ax.set_ylabel("Predicted {}".format(
                    devmodel.Data.columns[-(6 - i)].split("<")[0]),
                    wrap=True, fontsize=5)
                ax.set_xlabel("Actual {}".format(
                    devmodel.Data.columns[-(6 - i)].split("<")[0]),
                    wrap=True, fontsize=5)
            plt.xlim(minval, maxval)
            plt.ylim(minval, maxval)
            ax.grid()
        plt.tight_layout()
    return plt
