import math
import os
import glob
import numpy as np
import zarr
from zarr.storage import DirectoryStore
import dask.array as da
from ome_zarr.writer import write_image, write_multiscale, write_multiscale_labels
from ome_zarr.io import parse_url
from numcodecs import Zstd, Blosc, LZ4
from utils.utils_image import load_image, normalize, normalize_std, normalize_std_dask
from utils.utils_preprocess import image_crop_pad
from dask.diagnostics import ProgressBar

def write_ome_pyramid(image_group, image_pyramid, label_pyramid, chunk_size=(648, 648, 648), cname='lz4'):

    # Define the chunk sizes for each level
    chunk_sizes = [np.array(chunk_size) // (2**i) for i in range(len(image_pyramid))]
    print("Chunk sizes: ", chunk_sizes)

    # Define storage options for each level
    # Compressions: LZ4(), Zstd(level=3)
    storage_opts = [
        {"chunks": chunk_sizes[i], "compression": Blosc(cname=cname, clevel=3, shuffle=Blosc.BITSHUFFLE)}
        for i in range(len(image_pyramid))
    ]

    with ProgressBar(dt=1.0):
        # Write the image data to the Zarr group
        write_multiscale(
                image_pyramid,
                group=image_group,
                axes=["z", "y", "x"],
                storage_options=storage_opts
            )

    with ProgressBar(dt=1.0):
        if label_pyramid is not None:
            # Now write the label pyramid under /volume/labels/mask/
            write_multiscale_labels(
                label_pyramid,
                group=image_group,
                name="mask",
                axes=["z", "y", "x"],
                storage_options=storage_opts
            )

    print("Done writing multiscale data to OME-Zarr group")


def write_ome_datasample(out_name,
                         HR_paths,
                         LR_paths,
                         REG_paths,
                         HR_chunks,
                         LR_chunks,
                         REG_chunks,
                         HR_split_indices=(),
                         LR_split_indices=(),
                         REG_split_indices=(),
                         split_axis=0,
                         compression='lz4',
                         norm_method=None):

    """
    We need these images:
        ome.zarr
        - HR
            - fixed_scale_1.nii.gz
            - fixed_scale_2.nii.gz
            - fixed_scale_4.nii.gz
            - fixed_scale_8.nii.gz
        - LR
            - moving_scale_1.nii.gz
            - moving_scale_2.nii.gz
            - moving_scale_4.nii.gz
        - REG
            - <sample_name>.nii.gz
            - <sample_name>_scale_2.nii.gz
    """

    if len(HR_paths) == 0:
        raise ValueError("HR image paths are required and cannot be empty.")

    write_ome_group(image_paths=HR_paths,
                    out_name=out_name,
                    group_name='HR',
                    split_axis=split_axis,
                    split_indices=HR_split_indices,
                    chunks=HR_chunks,
                    compression=compression,
                    norm_method=norm_method)

    if len(LR_paths) == 0:
        print("No LR image paths provided, skipping LR group.")
    else:
        write_ome_group(image_paths=LR_paths,
                        out_name=out_name,
                        group_name='LR',
                        split_axis=split_axis,
                        split_indices=LR_split_indices,
                        chunks=LR_chunks,
                        compression=compression,
                        norm_method=norm_method)

    if len(REG_paths) == 0:
        print("No REG image paths provided, skipping REG group.")
    else:
        write_ome_group(image_paths=REG_paths,
                        out_name=out_name,
                        group_name='REG',
                        split_axis=split_axis,
                        split_indices=REG_split_indices,
                        chunks=REG_chunks,
                        compression=compression,
                        norm_method=norm_method)

    return 0


def write_ome_group(image_paths, out_name, group_name='HR', split_axis=0, split_indices=(), chunks=(160, 160, 160), compression='lz4', norm_method=None):

    if image_paths is None:
        raise ValueError("Image paths are required and cannot be empty.")

    # Load the image pyramid
    pyramid_splits = load_image_pyramid_splits(image_paths,
                                               split_axis,
                                               split_indices,
                                               dtype=np.float32,
                                               norm_method=norm_method)

    out_path = out_name

    if split_indices:
        # Create file name for each split index
        if ".zarr" in out_path:
            out_paths = [out_path.replace(".zarr", f"_{i}.zarr") for i in range(len(split_indices) + 1)]
        else:
            out_paths = [f"{out_path}_{i}.zarr" for i in range(len(split_indices) + 1)]
    else:
        if ".zarr" not in out_path:
            out_path = f"{out_path}.zarr"
        out_paths = [out_path]

    for i in range(len(split_indices) + 1):

        out_path = out_paths[i]
        print(f"Writing OME-Zarr data sample to {out_path}, split index {i}/{len(split_indices)}")

        # Create/open a Zarr array in write mode
        store = parse_url(out_path, mode="w").store
        # store = DirectoryStore(file_path)
        root = zarr.group(store=store)

        if os.path.exists(os.path.join(out_path, group_name)):
            print(f"Group {group_name} already exists in {out_path}. Skipping...")
            continue
        else:
            # Create image group for the volume
            image_group = root.create_group(group_name)

        write_ome_pyramid(
            image_group=image_group,
            image_pyramid=pyramid_splits[i],
            label_pyramid=None,  # No labels
            chunk_size=chunks,
            cname=compression  # Compression codec
        )

        print(f"Done writing OME-Zarr data to {out_path}/{group_name}")

    return 0


def load_image_pyramid(image_paths, dtype=np.float32, normalize=True):
    """
    Load a pyramid of images from given paths.
    If mask_paths are provided, apply the masks to the images.
    """
    pyramid = []
    for i, image_path in enumerate(image_paths):
        # Load image
        image = load_image(image_path, dtype=dtype)
        image = np.ascontiguousarray(image)

        if normalize:
            image = normalize_std(image, standard_deviations=3, mode='rescale')

        pyramid.append(image)

    return pyramid


def load_image_pyramid_splits(image_paths, split_axis=0, split_indices=(), dtype=np.float32, norm_method=None):

    # Load image pyramid
    pyramid = []
    for i, image_path in enumerate(image_paths):
        # Load image
        print(f"Loading image: {os.path.basename(image_path)}")
        image = load_image(image_path, dtype=dtype, as_contiguous=True, as_dask_array=True)

        if norm_method is None:
            print(f"Skipping normalization for image: {os.path.basename(image_path)}")
        elif norm_method == "min_max":
            print(f"Normalizing image: {os.path.basename(image_path)} using min-max normalization")
            image = normalize(image, global_min=None, global_max=None, dtype=image.dtype)
        elif norm_method == "std":
            print(f"Normalizing image: {os.path.basename(image_path)} to +/- 3 standard deviations")
            image = normalize_std(image, standard_deviations=3, mode='rescale')
        else:
            raise ValueError(f"Unsupported normalization method: {norm_method}")

        pyramid.append(image)

    if len(split_indices) == 0:
        # If no splitting is required, return the pyramid as is
        return [pyramid]

    # Split the pyramid images into num_splits along the specified axis
    pyramid_splits = [[] for _ in range(len(split_indices) + 1)]
    for i, image in enumerate(pyramid):
        # Check if the image is a Dask array
        if isinstance(image, da.Array):
            # TODO: Handle Dask arrays
            raise NotImplementedError("Dask array splitting is not implemented yet.")
        elif isinstance(image, np.ndarray):
            print(f"Splitting pyramid image: {i} along axis {split_axis} with indices {list(np.array(split_indices) // 2**i)}")
            splits = np.array_split(image, np.array(split_indices) // 2**i, axis=split_axis)

        for i, split in enumerate(splits):
            pyramid_splits[i].append(split)

    return pyramid_splits


if __name__ == "__main__":

    sample_name = "Femur_01_80kV"
    project_path = "../../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Femur_01"

    HR_paths = glob.glob(os.path.join(project_path, "fixed_scale_*.nii.gz"))
    LR_paths = glob.glob(os.path.join(project_path, "moving_scale_*.nii.gz"))
    REG_paths = glob.glob(os.path.join(project_path, f"{sample_name}*.nii.gz"))

    out_name = os.path.join(project_path, "Femur_01_ome")

    write_ome_datasample(out_name=out_name,
                         HR_paths=HR_paths,
                         LR_paths=LR_paths,
                         REG_paths=REG_paths,
                         HR_chunks=(160, 160, 160),
                         LR_chunks=(120, 120, 120),
                         REG_chunks=(40, 40, 40),
                         HR_split_indices=(320, ),
                         LR_split_indices=(480, ),
                         REG_split_indices=(480, ),
                         split_axis=0,
                         compression='lz4')

    print("Done writing OME-Zarr data sample")