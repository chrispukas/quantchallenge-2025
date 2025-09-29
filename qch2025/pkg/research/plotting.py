import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(x: list[float],
                y: list[float],
                title: str = "Title",
                x_label: str = "X Axis",
                y_label: str = "Y Axis") -> None:
    
    plt.scatter(x, y)

    plt.title(label=title)
    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)

    plt.ylim([0, 0.02])

    plt.show()

def plot_line(x: list[float],
                y: list[float],
                title: str = "Title",
                x_label: str = "X Axis",
                y_label: str = "Y Axis") -> None:
    
    plt.plot(x, y)

    plt.title(label=title)
    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)

    plt.show()


def get_rsq(y_pred: list, y_true: list):
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - rss/tss