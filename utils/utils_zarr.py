import math
import os
import glob
import numpy as np
import zarr
from zarr.storage import LocalStore
from skimage.exposure import match_histograms
import dask.array as da
from ome_zarr.writer import write_image, write_multiscale, write_multiscale_labels, write_multiscales_metadata, write_label_metadata
from ome_zarr.io import parse_url
from numcodecs import Zstd, Blosc, LZ4
from zarr.codecs import BytesCodec, BloscCodec, BloscCname, BloscShuffle
from PIL import Image

from utils.utils_image import load_image, normalize, normalize_std, normalize_std_dask, match_histogram_3d_continuous_sampled, compare_histograms
from utils.utils_preprocess import image_crop_pad
from dask.diagnostics import ProgressBar
from utils.utils_plot import viz_slices, viz_orthogonal_slices, viz_multiple_images
from utils.utils_nifti import write_nifti


def write_ome_metadata(group, num_levels, scale=2):
    # coordinate transforms are currently experimental in ome-zarr, see:
    # https://ngff-spec.readthedocs.io/en/latest/#affine-md
    datasets = [
        {"path": f"{lvl}",
         "coordinateTransformations": [
             {"type": "scale", "scale": [scale**lvl, scale**lvl, scale**lvl]}]} for lvl in range(num_levels)
         ]

    write_multiscales_metadata(
        group=group,
        datasets=datasets,
        axes=["z", "y", "x"],
    )



def create_ome_group(path, group_name='HR', pyramid_depth=4, scale=2, **kwargs):

    # Create/open a Zarr array in write mode
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)

    out_path = path
    if os.path.exists(os.path.join(path, group_name)):
        print(f"Group {group_name} already exists in {out_path}. Skipping...")
        image_group = None
    else:
        # Create image group for the volume
        image_group = root.create_group(group_name)

        write_ome_metadata(group=image_group, num_levels=pyramid_depth, scale=scale)
        print(f"Created OME-Zarr group at {os.path.basename(out_path)}/{group_name}")

    return store, image_group

# def write_ome_pyramid(image, store, group_name, pyramid_depth=4, scale=2):
#     # Write moving image ome-zarr level 0
#     pyramid = [image]
#     for level in range(pyramid_depth):
#         with ProgressBar(dt=1):
#             print(f"Writing moving pyramid level {level}...")
#             da.to_zarr(pyramid[level],
#                        url=store,
#                        component=f"{group_name}/{level}",
#                        overwrite=True,
#                        zarr_format=3)
#
#         pyramid[level] = da.from_zarr(store.root, component=f"{group_name}/{level}")
#
#         if level < pyramid_depth - 1:
#             down = da.coarsen(np.mean, pyramid[level], {0: 2, 1: 2, 2: 2}, trim_excess=True).astype(image.dtype)
#
#             # apply mask
#             down = da.where(mask_pyramid[level + 1].astype(bool), down, 0)
#             pyramid.append(down)
#
#     image = pyramid[0]  # refresh full resolution mask


# # Test
# moving_ome_path = os.path.join(moving_out_path, args.moving_out_name + "_ome.zarr")
# # Create/open a Zarr array in write mode
# store = parse_url(moving_ome_path, mode="w").store
# # store = zarr.storage.LocalStore(moving_ome_path)
# root = zarr.group(store=store)
#
# group_name = "LR"
# out_path = moving_ome_path
# if os.path.exists(os.path.join(moving_ome_path, group_name)):
#     print(f"Group {group_name} already exists in {out_path}. Skipping...")
# else:
#     # Create image group for the volume
#     image_group = root.create_group(group_name)
#
#     write_ome_metadata(group=image_group, num_levels=args.moving_pyramid_depth, scale=2)
#
#     with ProgressBar(dt=1):
#         da.to_zarr(moving,
#                    url=store,
#                    component=f"{group_name}/0",
#                    overwrite=True,
#                    zarr_format=3)
#
#     moving = da.from_zarr(moving_ome_path, chunks=moving.chunksize)


def write_ome_level(image, store, group_name, level=0, chunk_size=None, cname='lz4', clevel=3):

    if chunk_size is not None:
        if chunk_size != image.chunksize:
            image = image.rechunk(chunk_size)

    codecs = (
        BytesCodec(),
        BloscCodec(
            cname=BloscCname[cname],
            clevel=clevel,
            shuffle=BloscShuffle.shuffle
        )
    )

    component = f"{group_name}/{level}"

    # if already exists, skip
    if component not in store:
        with ProgressBar(dt=1):
            print(f"Writing OME level to {group_name}/{level}")

            # Calls to zarr.api.asynchronous.create under the hood, currently shards not supported...
            # https://zarr.readthedocs.io/en/stable/api/zarr/api/asynchronous/#zarr.api.asynchronous.create
            da.to_zarr(image,
                       url=store,
                       component=component,
                       overwrite=True,
                       zarr_format=3,
                       codecs=codecs,
                       )
    else:
        print(f"OME level {group_name}/{level} already exists, skipping write.")

    # Reload the written image level (clears dask graph)
    image = da.from_zarr(store, component=f"{group_name}/{level}")

    return image



def write_ome_pyramid(image_group, image_pyramid, label_pyramid, chunk_size=(648, 648, 648), shard_size=None, cname='lz4'):

    # Define the chunk sizes for each level
    #chunk_shapes = [np.array(chunk_size) // (2**i) for i in range(len(image_pyramid))]
    chunk_shapes = [np.array(chunk_size) for _ in range(len(image_pyramid))]
    print("Chunk shapes: ", chunk_shapes)

    # Define the shard sizes for each level
    shard_shapes = None
    if shard_size is not None:
        shard_shapes = [chunk_shapes[i] * np.array(shard_size) * (2 ** i) for i in range(len(image_pyramid))]
        print("Shard shapes: ", shard_shapes)

    # Define storage options for each level
    # Compressions: LZ4(), Zstd(level=3)
    storage_opts = [
        {
            "chunks": chunk_shapes[i].tolist(),
            "compressor": BloscCodec(cname=BloscCname[cname], clevel=3, shuffle=BloscShuffle.bitshuffle)
        }
        for i in range(len(image_pyramid))
    ]

    if shard_shapes is not None:
        for i in range(len(storage_opts)):
            storage_opts[i]["shards"] = shard_shapes[i].tolist()

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
                         HR_mask_paths,
                         LR_paths,
                         LR_mask_paths,
                         REG_paths,
                         REG_mask_paths,
                         HR_chunks,
                         LR_chunks,
                         REG_chunks,
                         HR_split_indices=(),
                         LR_split_indices=(),
                         REG_split_indices=(),
                         split_axis=0,
                         compression='lz4',
                         match_REG=True):

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

    hr_ref_dict = write_ome_group_resmatch(image_paths=HR_paths,
                                           mask_paths=HR_mask_paths,
                                           out_name=out_name,
                                           group_name='HR',
                                           split_axis=split_axis,
                                           split_indices=HR_split_indices,
                                           chunks=HR_chunks,
                                           compression=compression,
                                           return_levels=[2, 3])

    if len(LR_paths) == 0:
        print("No LR image paths provided, skipping LR group.")
    else:
        write_ome_group_resmatch(image_paths=LR_paths,
                        mask_paths=LR_mask_paths,
                        out_name=out_name,
                        group_name='LR',
                        split_axis=split_axis,
                        split_indices=LR_split_indices,
                        chunks=LR_chunks,
                        compression=compression)

    if len(REG_paths) == 0:
        print("No REG image paths provided, skipping REG group.")
    else:
        write_ome_group_resmatch(image_paths=REG_paths,
                        mask_paths=REG_mask_paths,
                        out_name=out_name,
                        group_name='REG',
                        split_axis=split_axis,
                        split_indices=REG_split_indices,
                        chunks=REG_chunks,
                        compression=compression,
                        reference_val_dict=hr_ref_dict,
                        match_slices=match_REG)

    return 0


def write_ome_group(image_paths, mask_paths=None, out_name="", group_name='HR', split_axis=0, split_indices=(), chunks=(160, 160, 160), compression='lz4', reference_vals=None):

    if image_paths is None:
        raise ValueError("Image paths are required and cannot be empty.")

    # Load the image pyramid
    pyramid = load_image_pyramid(image_paths, dtype=np.float32)
    print("Image pyramid shapes:", [img.shape for img in pyramid])

    mask_pyramid = None
    if mask_paths is not None:
        mask_pyramid = load_image_pyramid(mask_paths, dtype=np.uint8)
        print("Mask pyramid shapes:", [img.shape for img in mask_pyramid]) # ref_mask = load_image("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Cardboard_A/fixed_scale_1_mask.nii.gz")

    # match histograms
    start_idx = 0
    if reference_vals is None:
        reference_vals = pyramid[0]  # use first image as reference
        start_idx = 1  # normalize remaining images based on first image
        if mask_pyramid is not None:
            reference_mask = mask_pyramid[0].astype(bool)
            reference_vals = reference_vals[reference_mask]

    for i in range(start_idx, len(pyramid)):
        print(f"Matching histogram pyramid level {i}...")
        source_vals = pyramid[i]

        # viz_slices(pyramid[i], [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)

        if mask_pyramid is not None:
            source_mask = mask_pyramid[i].astype(bool)
            matched_vals = match_histograms(source_vals[source_mask], reference_vals)
            source_vals[source_mask] = matched_vals
            pyramid[i] = source_vals
        else:
            matched_vals = match_histograms(source_vals, reference_vals)
            pyramid[i] = matched_vals

        # viz_slices(pyramid[i], [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)

    ######
    # matched = match_histograms(pyramid[2], reference_image)
    # viz_slices(pyramid[2], [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)
    # viz_slices(matched, [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)
    # viz_slices(reference_image, [10 * 4, 20 * 4, 30 * 4], savefig=False, vmin=0, vmax=1, axis=0)
    # compare_histograms(pyramid[2], reference_image)
    ######

    # Split pyramid (if needed)
    pyramid_splits = split_pyramid(pyramid, split_axis, split_indices)

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

    return reference_vals


def write_ome_group_resmatch(image_paths, mask_paths=None, out_name="", group_name='HR', split_axis=0, split_indices=(), chunks=(160, 160, 160), compression='lz4', reference_val_dict={}, return_levels=(), match_slices=False):

    if image_paths is None:
        raise ValueError("Image paths are required and cannot be empty.")

    # Load the image pyramid
    pyramid = load_image_pyramid(image_paths, dtype=np.uint16)
    print("Image pyramid shapes:", [img.shape for img in pyramid])

    if group_name != "REG":
        viz_slices(pyramid[-2], [10, 20, 30], savefig=True, vmin=0, vmax=65535, axis=0, save_dir="", title=out_name + f"_{group_name}_scale_{len(pyramid) - 2}_raw")
        viz_slices(pyramid[-1], [10, 20, 30], savefig=True, vmin=0, vmax=65535, axis=0, save_dir="", title=out_name + f"_{group_name}_scale_{len(pyramid) - 1}_raw")

        # save single slices before matching
        image = Image.fromarray(pyramid[2][100].astype(np.uint16))
        image.save(f"figures/{group_name}_{100}_matched.png")

    mask_pyramid = None
    if mask_paths is not None:
        mask_pyramid = load_image_pyramid(mask_paths, dtype=np.uint8)
        print("Mask pyramid shapes:", [img.shape for img in mask_pyramid]) # ref_mask = load_image("../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Cardboard_A/fixed_scale_1_mask.nii.gz")

    return_dict = {}
    if group_name == "HR":
        for level in return_levels:
            if level < len(pyramid):
                if mask_pyramid is not None:
                    source_mask = mask_pyramid[level].astype(bool)
                    source_vals = np.where(source_mask, pyramid[level], np.nan)
                else:
                    source_vals = pyramid[level]
                return_dict[level] = source_vals

    if match_slices:
        # match histograms for REG group
        for i, level in enumerate(reference_val_dict):

            # Get source
            if mask_pyramid is not None:
                source_mask = mask_pyramid[i].astype(bool)
                source_vals = np.where(source_mask, pyramid[i], np.nan)
            else:
                source_vals = pyramid[i]

            # Get reference
            reference_vals = reference_val_dict[level]
            print(f"Matching histogram level {i} with reference level {level}...")

            viz_slices(pyramid[i], [10, 20, 30], savefig=True, vmin=0, vmax=65535, axis=0, save_dir="", title=out_name + f"_{group_name}_scale_{i}_raw")

            if i == 0:
                # save single slices before matching
                image = Image.fromarray(pyramid[0][100].astype(np.uint16))
                image.save(f"figures/{group_name}_{100}_unmatched.png")

            if mask_pyramid is not None:
                for slice_idx in range(source_vals.shape[0]):
                    if slice_idx % 100 == 0:
                        print(f"Matching slice {slice_idx}/{source_vals.shape[0]}")
                    matched_slice = match_histograms(source_vals[slice_idx], reference_vals[slice_idx])
                    matched_slice = np.nan_to_num(matched_slice, nan=0)  # fill nans with 0
                    pyramid[i][slice_idx] = matched_slice
            else:
                for slice_idx in range(source_vals.shape[0]):
                    if slice_idx % 100 == 0:
                        print(f"Matching slice {slice_idx}/{source_vals.shape[0]}")
                    matched_slice = match_histograms(source_vals[slice_idx], reference_vals[slice_idx])
                    pyramid[i][slice_idx] = matched_slice

            viz_slices(pyramid[i], [10, 20, 30], savefig=True, vmin=0, vmax=65535, axis=0, save_dir="", title=out_name + f"_{group_name}_scale_{i}_matched")

            if i == 0:
                # save single slices before matching
                image = Image.fromarray(pyramid[0][100].astype(np.uint16))
                image.save(f"figures/{group_name}_{100}_matched.png")
                np.save(f"figures/source_vals_{100}.npy", source_vals[100])
                np.save(f"figures/reference_vals_{100}.npy", reference_vals[100])
                np.save(f"figures/matched_vals_{100}.npy",  pyramid[i][100])

            # Optionally, save matched image for verification
            write_nifti(pyramid[i], output_path=out_name + f"_{group_name}_matched_scale_{i}.nii.gz", dtype=np.uint16)

    ######
    # matched = match_histograms(pyramid[2], reference_image)
    # viz_slices(pyramid[2], [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)
    # viz_slices(matched, [10, 20, 30], savefig=False, vmin=0, vmax=1, axis=0)
    # viz_slices(reference_image, [10 * 4, 20 * 4, 30 * 4], savefig=False, vmin=0, vmax=1, axis=0)
    # compare_histograms(pyramid[2], reference_vals)
    ######

    # Split pyramid (if needed)
    pyramid_splits = split_pyramid(pyramid, split_axis, split_indices)

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

        # pyramid_splits[i] = [da.array(img) for img in pyramid_splits[i]] # REMOVE THIS

        write_ome_pyramid(
            image_group=image_group,
            image_pyramid=pyramid_splits[i],
            label_pyramid=None,  # No labels
            chunk_size=chunks,
            shard_size=None,  # None if no shards
            cname=compression,  # Compression codec
        )

        print(f"Done writing OME-Zarr data to {out_path}/{group_name}")

    return return_dict


def load_image_pyramid(image_paths, dtype=np.uint16, nifti_backend="nibabel"):
    """
    Load a pyramid of images from given paths.
    If mask_paths are provided, apply the masks to the images.
    """
    pyramid = []
    for i, image_path in enumerate(image_paths):
        print(f"Loading pyramid level {i}...")
        image = load_image(image_path, dtype=dtype, as_contiguous=True, nifti_backend=nifti_backend)
        pyramid.append(image)

    return pyramid


def split_pyramid(pyramid, split_axis=0, split_indices=()):

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

def load_image_pyramid_splits(image_paths, split_axis=0, split_indices=(), dtype=np.float32, norm_method=None):

    # Load image pyramid
    pyramid = []
    for i, image_path in enumerate(image_paths):
        # Load image
        print(f"Loading image: {os.path.basename(image_path)}")
        image = load_image(image_path, dtype=dtype, as_contiguous=True, as_dask_array=False)

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