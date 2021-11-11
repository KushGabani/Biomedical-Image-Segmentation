import random
import numpy as np
import matplotlib.pyplot as plt


def get_random_samples(_samples=1, phase="test"):
    data = np.load("./preprocessed_data.npz")
    X_train, y_train = data['X_' + phase], data['y_' + phase]
    pairs = []
    for i in range(0, _samples):
        i = random.randint(0, len(X_train))
        pairs.append((X_train[i], y_train[i]))

    return pairs


def plot_data_pair(X, y):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(X, cmap='gray')
    plt.subplot(122)
    plt.imshow(y, cmap='gray')
    plt.show()


def plot_random_samples():
    X, y = get_random_samples(1)[0]
    plot_data_pair(X, y)


def plot_n_pairs(n=3):
    pairs = get_random_samples(n)
    fig, axes = plt.subplots(n, 2)
    for i in range(n):
        X, y = pairs[i]
        axes[i][0].imshow(X, cmap='gray')
        axes[i][1].imshow(y, cmap='gray')
        for j in range(2):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

    plt.show()


if __name__ == "__main__":
    plot_random_samples()
    plot_n_pairs()
