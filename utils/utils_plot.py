import os
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_itk import itk_checkerboard

def viz_slices(array, slice_indices, title="output", axis=0, savefig=True, save_dir="figures", vmin=None, vmax=None):
    # if slice_indices is int, convert to list
    if isinstance(slice_indices, int):
        slice_indices = [slice_indices]

    plt.figure(figsize=(8*len(slice_indices), 8))

    for slice in slice_indices:
        plt.subplot(1, len(slice_indices), slice_indices.index(slice) + 1)
        if axis == 1:
            plt.imshow(array[:, slice, :], vmin=vmin, vmax=vmax)
        elif axis == 2:
            plt.imshow(array[:, :, slice], vmin=vmin, vmax=vmax)
        else:
            # Default to axis 0
            plt.imshow(array[slice, :, :], vmin=vmin, vmax=vmax)
        plt.title(f"Slice: {slice}, axis: {axis}", fontsize=16)

    if savefig:
        plt.savefig(os.path.join(save_dir, title + ".png"), dpi=300, bbox_inches='tight')
        print(f"Saved figure as {title}")
    else:
        plt.show()

def viz_orthogonal_slices(array, slice_indices, title="output", savefig=True, save_dir="figures", vmin=None, vmax=None):

    # if slice_indices is int, convert to list
    if isinstance(slice_indices, int):
        slice_indices = [slice_indices]

    plt.figure(figsize=(8 * len(slice_indices), 8*3))

    plot_count = 1
    for axis in range(3):
        for slice in slice_indices:
            plt.subplot(3, len(slice_indices), plot_count)
            if axis == 1:
                plt.imshow(array[:, slice, :], vmin=vmin, vmax=vmax)

            elif axis == 2:
                plt.imshow(array[:, :, slice], vmin=vmin, vmax=vmax)
            else:
                # Default to axis 0
                plt.imshow(array[slice, :, :], vmin=vmin, vmax=vmax)

            plt.title(f"Slice: {slice}, axis: {axis}", fontsize=16)
            plt.axis('off')
            plot_count += 1

    if savefig:
        path = os.path.join(save_dir, title + ".png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved figure {path}")
    else:
        plt.show()


def viz_multiple_images(image_list, slice_indices, title=None, axis=0, savefig=True, save_dir="figures", vmin=None, vmax=None):

    plt.figure(figsize=(8*len(slice_indices), 8*len(image_list)))

    plot_count = 1
    for image in image_list:
        for slice in slice_indices:
            plt.subplot(len(image_list), len(slice_indices), plot_count)
            if axis == 1:
                plt.imshow(image[:, slice, :], cmap='gray', vmin=vmin, vmax=vmax)
            elif axis == 2:
                plt.imshow(image[:, :, slice], cmap='gray', vmin=vmin, vmax=vmax)
            else:
                # Default to axis 0
                plt.imshow(image[slice, :, :], cmap='gray', vmin=vmin, vmax=vmax)
            plt.title(f"Slice: {slice}, axis: {axis}", fontsize=16)
            plt.axis('off')
            plot_count += 1

    if savefig:
        plt.savefig(os.path.join(save_dir, title + ".png"), dpi=300, bbox_inches='tight')
        print(f"Saved figure as {title}")
    else:
        plt.show()


def viz_registration(fixed_image, moving_image, slice_indices, title=None, axis=0, checker_pattern=(20, 20, 20), savefig=True, save_dir="figures", vmin=None, vmax=None):
    plt.figure(figsize=(8 * len(slice_indices), 8 * 2))

    # Create checkerboard image
    checker = itk_checkerboard(fixed_image, moving_image, checker_pattern)

    plot_count = 1
    for slice in slice_indices:
        # Plot the difference between fixed and moving images
        plt.subplot(2, len(slice_indices), plot_count)
        if axis == 1:
            plt.imshow(fixed_image[:, slice, :] - moving_image[:, slice, :], vmin=vmin, vmax=vmax)
        elif axis == 2:
            plt.imshow(fixed_image[:, :, slice] - moving_image[:, :, slice], vmin=vmin, vmax=vmax)
        else:
            # Default to axis 0
            plt.imshow(fixed_image[slice, :, :] - moving_image[slice, :, :], vmin=vmin, vmax=vmax)
        plt.title(slice)
        plt.axis('off')
        plot_count += 1

    plot_count = len(slice_indices) + 1
    for slice in slice_indices:
        # Plot checkerboard image
        plt.subplot(2, len(slice_indices), plot_count)
        if axis == 1:
            plt.imshow(checker[:, slice, :], vmin=vmin, vmax=vmax)
        elif axis == 2:
            plt.imshow(checker[:, :, slice], vmin=vmin, vmax=vmax)
        else:
            # Default to axis 2
            plt.imshow(checker[slice, :, :], vmin=vmin, vmax=vmax)
        plt.title(slice)
        plt.axis('off')
        plot_count += 1

    if savefig:
        plt.savefig(os.path.join(save_dir, title + ".png"), dpi=300, bbox_inches='tight')
        print(f"Saved figure as {title}.png")
    else:
        plt.show()


