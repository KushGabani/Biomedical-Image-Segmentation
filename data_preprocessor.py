import os
import tifffile as tif
import numpy as np


def preprocess(directory, isX=False):
    data = []
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            image = tif.imread(os.path.join(directory, file)).astype(float)
            if isX:
                image /= 255.
            data.append(image)

    return np.array(data)


def load_data(npz_file):
    data = np.load(npz_file)
    return data['images'], data['masks']


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

    np.savez_compressed("./preprocessed_data.npz", images=images, masks=masks)
    print("Preprocessing completed! Data saved to ./preprocessed_data.npz.")
