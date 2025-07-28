
import os
import glob
import numpy as np
import zarr
from zarr.storage import DirectoryStore
import dask.array as da
from ome_zarr.writer import write_image, write_multiscale, write_multiscale_labels
from ome_zarr.io import parse_url
from numcodecs import Zstd, Blosc, LZ4
from utils.utils_image import load_image, normalize_std
from utils.utils_preprocess import image_crop_pad

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

    # Write the image data to the Zarr group
    write_multiscale(
            image_pyramid,
            group=image_group,
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )

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


def write_ome_datasample(out_name, HR_paths, LR_paths, REG_paths, HR_chunks, LR_chunks, REG_chunks, split_axis=0, num_split_sections=1, compression='lz4'):

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

    if HR_paths is None:
        raise ValueError("HR image paths are required and cannot be empty.")

    write_ome_group(image_paths=HR_paths,
                    out_name=out_name,
                    group_name='HR',
                    split_axis=split_axis,
                    num_split_sections=num_split_sections,
                    chunks=HR_chunks,
                    compression=compression)

    if len(LR_paths) == 0:
        print("No LR image paths provided, skipping LR group.")
    else:
        write_ome_group(image_paths=LR_paths,
                        out_name=out_name,
                        group_name='LR',
                        split_axis=split_axis,
                        num_split_sections=num_split_sections,
                        chunks=LR_chunks,
                        compression=compression)

    if len(REG_paths) == 0:
        print("No REG image paths provided, skipping REG group.")
    else:
        write_ome_group(image_paths=REG_paths,
                        out_name=out_name,
                        group_name='REG',
                        split_axis=split_axis,
                        num_split_sections=num_split_sections,
                        chunks=REG_chunks,
                        compression=compression)

    return 0


def write_ome_group(image_paths, out_name, group_name='HR', split_axis=0, num_split_sections=1, chunks=(160, 160, 160), compression='lz4'):

    if image_paths is None:
        raise ValueError("Image paths are required and cannot be empty.")

    # Load the image pyramid
    pyramid_splits = load_image_pyramid_splits(image_paths,
                                               split_axis,
                                               num_split_sections,
                                               dtype=np.float32,
                                               normalize=True)

    for i in range(num_split_sections):
        # Create the output path for each split
        if num_split_sections > 1:
            if ".zarr" in out_name:
                out_name = out_name.replace(".zarr", f"_{i}.zarr")
            else:
                out_path = f"{out_name}_{i}.zarr"
            print(f"Writing OME-Zarr data sample split {i} to {out_path}")
        else:
            if ".zarr" in out_name:
                out_path = out_name
            else:
                out_path = f"{out_name}.zarr"
            print(f"Writing OME-Zarr data sample to {out_path}")

        # Create/open a Zarr array in write mode
        store = parse_url(out_path, mode="w").store
        # store = DirectoryStore(file_path)
        root = zarr.group(store=store)

        if os.path.exists(os.path.join(out_path, group_name)):
            print(f"Group {group_name} already exists in {out_path}. Skipping...")
            return 0
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


def load_image_pyramid_splits(image_paths, split_axis=0, num_split_sections=1, dtype=np.float32, normalize=True):

    # Load image pyramid
    pyramid = []
    for i, image_path in enumerate(image_paths):
        # Load image
        image = load_image(image_path, dtype=dtype)
        image = np.ascontiguousarray(image)

        if normalize:
            image = normalize_std(image, standard_deviations=3, mode='rescale')

        pyramid.append(image)

    if num_split_sections <= 1:
        # If no splitting is required, return the pyramid as is
        return [pyramid]

    # Split the pyramid images into num_splits along the specified axis
    pyramid_splits = [] * num_split_sections
    for image in pyramid:
        splits = np.array_split(image, num_split_sections, axis=split_axis)

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
                         split_axis=0,
                         num_split_sections=1,
                         compression='lz4')

    print("Done writing OME-Zarr data sample")