import sys
import os
import tifffile as tif
import numpy as np


def create_patches_from_image(image, patch_size):
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    print("# of patches: {}".format(len(patches)))
    return patches


def save_patches(patches, directory, filename):
    for i, patch in enumerate(patches):
        filepath = os.path.join(directory, "{}_{}.tif".format(str(filename), str(i)))
        print("saving {}...".format(filepath))
        tif.imsave(filepath, patch)


def create_patches_from_dir(patch_size, DATASET_DIR="E:/Datasets/Mitochondria-Seg-Dataset/"):
    data_stack = []
    mask_stack = []

    for phase in ["training", "testing"]:
        sub_data_stack = tif.imread(DATASET_DIR + phase + ".tif")
        sub_mask_stack = tif.imread(DATASET_DIR + phase + "_groundtruth.tif")

        print("{} data stack shape: {}".format(phase, sub_data_stack.shape))
        print("{} label stack shape: {}".format(phase, sub_mask_stack.shape))
        data_stack.append(sub_data_stack)
        mask_stack.append(sub_mask_stack)

    data_stack = np.vstack((data_stack[0], data_stack[1]))
    mask_stack = np.vstack((mask_stack[0], mask_stack[1]))
    assert type(data_stack) == np.ndarray
    assert type(mask_stack) == np.ndarray
    print("Combined data stack shape: {}".format(data_stack.shape))
    print("Combined mask stack shape: {}".format(mask_stack.shape))

    dir_path = "./data"
    if not os.path.exists(dir_path + "/images"):
        os.mkdir(os.path.join(dir_path, "images"))
    if not os.path.exists(dir_path + "/masks"):
        os.mkdir(os.path.join(dir_path, "masks"))

    for x, stack in enumerate([data_stack, mask_stack]):
        count = 0
        for index in range(stack.shape[0]):
            large_img = stack[index]
            print("current stack image ({}): {}".format(phase, index))
            patches = create_patches_from_image(large_img, patch_size)
            save_patches(
                patches,
                os.path.join(dir_path, ("images" if x == 0 else "masks")),
                count
            )
            count += 1

        print("Total patch data created: {}".format(len(os.listdir(os.path.join(dir_path, "images")))))
        print("Total patch masks created: {}".format(len(os.listdir(os.path.join(dir_path, "masks")))))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_patches.py <dataset_dir> <patch_size>\nDefault Patch Size is 256 if not provided")
        exit(1)

    DATASET_DIR = str(sys.argv[1])

    if len(sys.argv) == 3:
        PATCH_SIZE = int(sys.argv[2])
    else:
        PATCH_SIZE = 256

    create_patches_from_dir(PATCH_SIZE, DATASET_DIR)
