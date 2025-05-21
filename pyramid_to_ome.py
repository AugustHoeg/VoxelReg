import os
import nibabel as nib
import numpy as np
import argparse
import zarr
from ome_zarr.writer import write_multiscale, write_multiscale_labels
from ome_zarr.io import parse_url
import dask.array as da
from numcodecs import Zstd, Blosc, LZ4
from utils.utils_zarr import write_ome_pyramid

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
        print(f"Reading image: {path}")
        image = nib.load(path).get_fdata()
        image_pyramid.append(image)

    # Read label pyramid
    if label_pyramid_paths is not None:
        label_pyramid = []
        for path in label_pyramid_paths:
            print(f"Reading label: {path}")
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

    # Open target Zarr group
    zarr_path = os.path.join(out_path, out_name)
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store)

    # Step 1: Create a multiscale image group
    image_group = root.create_group("HR")

    # Step 2: Write multiscale data to the image group with per-scale storage options
    write_ome_pyramid(image_group, image_pyramid, label_pyramid, chunk_size=(648, 648, 648))

    if args.registered_image_paths is not None:
        registered_image_paths = [os.path.join(sample_path, path) for path in args.registred_image_paths]
        print("registered image paths: ", registered_image_paths)

        registered_pyramid, _ = read_nifti_pyramid(registered_image_paths)

        registered_image_group = root.create_group("LR")

        storage_opts = [
            {"chunks": (162, 162, 162), "compressor": Blosc(cname='lz4', clevel=3, shuffle=Blosc.BITSHUFFLE)}
        ]

        write_multiscale(
            registered_pyramid,
            group=registered_image_group,
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )

    print("Done")