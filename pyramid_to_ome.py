import os
import nibabel as nib
import numpy as np
import argparse
import h5py
import zarr
from ome_zarr.writer import write_multiscale, write_multiscale_labels
from ome_zarr.io import parse_url
import dask.array as da
from numcodecs import Zstd, Blosc, LZ4
from utils.utils_zarr import write_ome_pyramid
from utils.utils_image import load_image, normalize
from dask.distributed import Client, LocalCluster

def read_pyramid(image_pyramid_paths, label_pyramid_paths=None, dataset_name=None, dtype=np.float16):

    image_pyramid = []
    for path in image_pyramid_paths:
        print(f"Reading image: {path}")
        volume = load_image(path, dtype=dtype, dataset_name=dataset_name)
        image_pyramid.append(volume)

    # Read label pyramid
    if label_pyramid_paths is not None:
        label_pyramid = []
        for path in label_pyramid_paths:
            print(f"Reading label: {path}")
            label_volume = load_image(path, dtype=dtype, dataset_name=dataset_name)
            label_pyramid.append(label_volume)
    else:
        label_pyramid = None

    return image_pyramid, label_pyramid


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
    else:
        label_pyramid = None

    return image_pyramid, label_pyramid


def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Read image/label pyramid and save as OME-Zarr.")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--image_paths", type=str, nargs='*', required=False, default=None, help="Path to images.")
    parser.add_argument("--label_paths", type=str, nargs='*', required=False, default=None, help="Path to labels.")
    parser.add_argument("--out_path", type=str, required=False, help="Path to the output file.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")
    parser.add_argument("--registered_image_paths", type=str, nargs='*', default=None, required=False, help="Path to registered images." )

    # chunk size for the OME-Zarr pyramid
    parser.add_argument("--chunk_size", type=int, nargs=3, default=(648, 648, 648), help="Chunk size for the OME-Zarr pyramid top.")
    parser.add_argument("--global_min_vals", type=float, nargs='*', default=None, help="Global minimum for normalization.")
    parser.add_argument("--global_max_vals", type=float, nargs='*', default=None, help="Global maximum for normalization.")
    parser.add_argument("--use_dask_cluster", default=False, help="Use Dask cluster for multiprocessing.")

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
        args.image_paths = [os.path.join(sample_path, path) for path in args.image_paths]
        print("image paths: ", args.image_paths)
    if args.label_paths is not None:
        args.label_paths = [os.path.join(sample_path, path) for path in args.label_paths]
        print("image paths: ", args.label_paths)
    if args.out_path is not None:
        out_path = os.path.join(sample_path, args.out_path)  # os.path.join(sample_path, args.out_name)
        print("Output path: ", out_path)
    if args.out_name is not None:
        out_name = args.out_name  # os.path.join(sample_path, args.out_name)
        print("Output name: ", out_name)

    # Read image and label pyramid
    #image_pyramid, label_pyramid = read_nifti_pyramid(args.image_paths, args.label_paths)
    image_pyramid, label_pyramid = read_pyramid(args.image_paths,
                                                args.label_paths,
                                                dataset_name='/exchange/data',
                                                dtype=np.float16)

    if args.use_dask_cluster:
        print("Using Dask cluster for multiprocessing...")
        # Step 1: Start Dask cluster (multiprocessing)
        cluster = LocalCluster(processes=True)
        client = Client(cluster)

    # Normalize the image pyramid if global min/max values are provided
    if args.global_min_vals or args.global_max_vals:
        for i in range(len(image_pyramid)):
            image_pyramid[i] = normalize(image_pyramid[i],
                                         global_min=args.global_min_vals[i],
                                         global_max=args.global_min_vals[i],
                                         dtype=np.float16)

    # Open target Zarr group
    zarr_path = os.path.join(out_path, out_name)
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store)

    # Step 1: Create a multiscale image group
    image_group = root.create_group("HR")

    # Step 2: Write multiscale data to the image group with per-scale storage options
    write_ome_pyramid(image_group, image_pyramid, label_pyramid, chunk_size=args.chunk_size)

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

    print("Conversion complete.")
    if args.use_dask_cluster:
        print("Closing Dask client and cluster...")
        client.close()
        cluster.close()

    print("Done")