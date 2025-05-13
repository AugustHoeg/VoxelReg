import os
import nibabel as nib
import numpy as np
import argparse
import zarr
from ome_zarr.writer import write_image, write_multiscale, write_multiscale_labels
from ome_zarr.scale import Scaler
from ome_zarr.io import parse_url
import dask.array as da
from numcodecs import Zstd, Blosc, LZ4

def read_nifti_pyramid(image_pyramid_paths, label_pyramid_paths=None):
    """
    Read the image and label pyramid.

    :param image_pyramid_paths:
    :param label_pyramid_paths:
    :return:
    """

    # Read image pyramid
    image_pyramid = []
    for path in image_pyramid_paths:
        image = nib.load(path).get_fdata()
        image_pyramid.append(image)

    # Read label pyramid
    if label_pyramid_paths is not None:
        label_pyramid = []
        for path in label_pyramid_paths:
            label = nib.load(path).get_fdata()
            label_pyramid.append(label)

    return image_pyramid, label_pyramid


def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Read image/label pyramid and save as OME-Zarr.")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--image_paths", type=str, nargs='*', required=False, help="Path to images.")
    parser.add_argument("--label_paths", type=str, nargs='*', required=False, help="Path to labels.")
    parser.add_argument("--out_path", type=str, required=False, help="Path to the output file.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")
    parser.add_argument("--registered_image_paths", type=str, nargs='*', default=None, required=False, help="Path to registered images." )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    if args.run_type == "HOME PC":
        project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    elif args.run_type == "DTU_HPC":
        project_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/"

    # Assign paths
    if args.sample_path is not None:
        sample_path = os.path.join(project_path, args.sample_path)
        print("Sample path: ", sample_path)
    if args.image_paths is not None:
        image_paths = [os.path.join(sample_path, path) for path in args.image_paths]
        print("image paths: ", image_paths)
    if args.label_paths is not None:
        label_paths = [os.path.join(sample_path, path) for path in args.label_paths]
        print("image paths: ", label_paths)
    if args.out_path is not None:
        out_path = os.path.join(sample_path, args.out_path)  # os.path.join(sample_path, args.out_name)
        print("Output path: ", out_path)
    if args.out_name is not None:
        out_name = args.out_name  # os.path.join(sample_path, args.out_name)
        print("Output name: ", out_name)


    # Read image and label pyramid
    image_pyramid, label_pyramid = read_nifti_pyramid(image_paths, label_paths)

    # Storage options for each level
    storage_opts = [
        {"chunks": (648, 648, 648), "compressor": Zstd(level=5)},
        {"chunks": (324, 324, 324), "compressor": Zstd(level=3)},
        {"chunks": (162, 162, 162), "compressor": Zstd(level=1)},
    ]

    # Open target Zarr group
    store = parse_url("ome_test.zarr", mode="w").store
    root = zarr.group(store=store)

    # Step 1: Create a multiscale image group
    image_group = root.create_group("volume")

    # Step 2: Write multiscale data to the image group with per-scale storage options
    write_multiscale(
        image_pyramid,
        group=image_group,
        axes=["z", "y", "x"],
        storage_options=storage_opts
    )

    # Step 3: Write the multiscale labels into a subpath under the image group
    # Now write the label pyramid under /volume/labels/mask/
    write_multiscale_labels(
        label_pyramid,
        group=image_group,
        name="mask",
        axes=["z", "y", "x"],
        storage_options=storage_opts
    )

    if args.registered_image_paths is not None:
        registered_image_paths = [os.path.join(sample_path, path) for path in args.registred_image_paths]
        print("registered image paths: ", registered_image_paths)

        registered_pyramid, _ = read_nifti_pyramid(registered_image_paths)

        registered_image_group = root.create_group("lowres")

        write_multiscale(
            registered_pyramid,
            group=registered_image_group,
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )

    print("Done")