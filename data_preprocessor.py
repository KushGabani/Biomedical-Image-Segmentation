import os
import gc
import tifffile as tif
import numpy as np
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split


def preprocess(directory, isX=False):
    data = []
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            image = tif.imread(os.path.join(directory, file)).astype(float)
            image /= 255.
            data.append(image)

    if isX:
        return np.expand_dims(normalize(np.array(data), axis=1), 3)
    else:
        return np.expand_dims((np.array(data)), 3) / 255.


def load_data(npz_file):
    """
    :param npz_file: filepath of the npz compressed data
    :returns X_train, X_test, y_train, y_test
    """
    data = np.load(npz_file)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']


if __name__ == "__main__":
    if not os.path.exists("./data"):
        print("Data patches does not exist, create patches (Refer ReadMe.md).\nStopping operation..")
        exit(1)
    elif not (os.path.exists("./data/images") or os.path.exists("./data/masks")):
        print("Patches not found. Run create_patches.py (Refer ReadMe.md).\nStopping operation..")
        exit(1)

    # Load images
    images = preprocess("./data/images/", isX=True)
    masks = preprocess("./data/masks/")

    print("image data shape: {}".format(images.shape))
    print("mask data shape: {}".format(masks.shape))

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.1, random_state=0)
    del images
    del masks
    gc.collect()

    np.savez_compressed("./preprocessed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Preprocessing completed! Data saved to ./preprocessed_data.npz.")
