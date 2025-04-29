import numpy as np
import matplotlib.pyplot as plt
from utils.utils_itk import itk_checkerboard

def viz_slices(array, slice_indices, title=None, axis=2, savefig=False):
    plt.figure(figsize=(8*len(slice_indices), 8))

    for slice in slice_indices:
        plt.subplot(1, len(slice_indices), slice_indices.index(slice) + 1)
        if axis == 0:
            plt.imshow(array[:, slice, :], vmin=0)
        elif axis == 1:
            plt.imshow(array[:, :, slice], vmin=0)
        else:
            # Default to axis 2
            plt.imshow(array[slice, :, :], vmin=0)
        plt.title(slice)

    if savefig:
        plt.savefig(f"figures/{title}.png", dpi=300, bbox_inches='tight')
        print(f"Saved figure as {title}.png")
    else:
        plt.show()


def viz_multiple_images(image_list, slice_indices, title=None, axis=2, savefig=False):

    plt.figure(figsize=(8*len(slice_indices), 8*len(image_list)))

    plot_count = 1
    for image in image_list:
        for slice in slice_indices:
            plt.subplot(len(image_list), len(slice_indices), plot_count)
            if axis == 0:
                plt.imshow(image[:, slice, :])
            elif axis == 1:
                plt.imshow(image[:, :, slice])
            else:
                # Default to axis 2
                plt.imshow(image[slice, :, :])
            plt.title(slice)
            plt.axis('off')
            plot_count += 1

    if savefig:
        plt.savefig(f"figures/{title}.png", dpi=300, bbox_inches='tight')
        print(f"Saved figure as {title}.png")
    else:
        plt.show()


def viz_registration(fixed_image, moving_image, slice_indices, title=None, axis=2, checker_pattern=(20, 20, 20), savefig=False):
    plt.figure(figsize=(8 * len(slice_indices), 8 * 2))

    # Create checkerboard image
    checker = itk_checkerboard(fixed_image, moving_image, checker_pattern)

    plot_count = 1
    for slice in slice_indices:
        # Plot the difference between fixed and moving images
        plt.subplot(2, len(slice_indices), plot_count)
        if axis == 0:
            plt.imshow(fixed_image[:, slice, :] - moving_image[:, slice, :])
        elif axis == 1:
            plt.imshow(fixed_image[:, :, slice] - moving_image[:, :, slice])
        else:
            # Default to axis 2
            plt.imshow(fixed_image[slice, :, :] - moving_image[slice, :, :])
        plt.title(slice)
        plt.axis('off')
        plot_count += 1

    plot_count = len(slice_indices) + 1
    for slice in slice_indices:
        # Plot checkerboard image
        plt.subplot(2, len(slice_indices), plot_count)
        if axis == 0:
            plt.imshow(checker[:, slice, :])
        elif axis == 1:
            plt.imshow(checker[:, :, slice])
        else:
            # Default to axis 2
            plt.imshow(checker[slice, :, :])
        plt.title(slice)
        plt.axis('off')
        plot_count += 1

    if savefig:
        plt.savefig(f"figures/{title}.png", dpi=300, bbox_inches='tight')
        print(f"Saved figure as {title}.png")
    else:
        plt.show()