import os
import zarr
import argparse
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

from utils.utils_zarr import create_ome_group, write_ome_level

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split volume and save")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--scan_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--out_ome_path", type=str, required=False, help="Path to the output file.")

    parser.add_argument("--chunk_size", type=int, nargs=3, default=(160, 160, 160), help="Size of each chunk (D, H, W).")
    parser.add_argument("--hr_split_indices", type=int, nargs='+', default=[480], help="Indices along the split axis where the volume will be split.")
    parser.add_argument("--lr_split_indices", type=int, nargs='+', default=[], help="Indices along the split axis where the volume will be split.")
    parser.add_argument("--split_axis", type=int, default=0, help="Axis along which to split the volume (0 for depth, 1 for height, 2 for width).")

    args = parser.parse_args()
    return args

def split_level(arr, base_indices, level, split_axis, num_files):
    """
    Split a single pyramid level (a dask array) along ``split_axis`` into
    ``num_files`` parts.

    ``base_indices`` are the split positions defined at level 0 of the *group*.
    For deeper pyramid levels the positions are scaled down by ``2**level`` so
    the cut planes stay spatially aligned across the pyramid.

    If ``base_indices`` is empty the whole level is replicated into every output
    file (e.g. LR when no lr_split_indices are given).
    """
    if len(base_indices) == 0:
        return [arr for _ in range(num_files)]

    # Scale the split positions for this pyramid level
    cuts = [idx // (2 ** level) for idx in base_indices]
    boundaries = [0] + list(cuts) + [arr.shape[split_axis]]

    parts = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        sl = [slice(None)] * arr.ndim
        sl[split_axis] = slice(start, stop)
        parts.append(arr[tuple(sl)])
    return parts


def build_split_indices_map(z, group_names, hr_split_indices, lr_split_indices, split_axis):
    """
    Build an explicit {group_name: base_split_indices} mapping.

    HR / HR_mask use hr_split_indices, LR / LR_mask use lr_split_indices, and
    REG (a lower-resolution copy of HR) derives its indices from the HR->REG
    shape ratio along ``split_axis`` (e.g. 1920/480 = 4 -> 480 becomes 120).
    """
    split_indices_map = {}
    for group_name in group_names:
        if group_name in ("HR", "HR_mask"):
            split_indices_map[group_name] = list(hr_split_indices)
        elif group_name in ("LR", "LR_mask"):
            split_indices_map[group_name] = list(lr_split_indices)
        elif group_name == "REG":
            factor = z["HR/0"].shape[split_axis] // z["REG/0"].shape[split_axis]
            print(f"Derived HR->REG downsample factor along axis {split_axis}: {factor}")
            split_indices_map[group_name] = [idx // factor for idx in hr_split_indices]
        else:
            raise ValueError(f"Unknown group '{group_name}': no split-index rule defined.")
    return split_indices_map

if __name__ == "__main__":

    args = parse_arguments()

    #### REMOVE THESE LINES
    if False:
        args.sample_path = "../3D_datasets/datasets/VoDaSuRe/Vertebrae_A/"
        args.scan_path = "Vertebrae_A_80kV_out_ome.zarr"
        args.out_ome_path = "Vertebrae_A_80kV_out_ome.zarr"
        args.chunk_size = (160, 160, 160)
        args.hr_split_indices = [480]
        args.lr_split_indices = []
        args.split_axis = 0
    ####

    zarr_path = os.path.join(args.sample_path, args.scan_path)
    out_ome_path = os.path.join(args.sample_path, args.out_ome_path)
    split_axis = args.split_axis
    hr_split_indices = args.hr_split_indices
    lr_split_indices = args.lr_split_indices

    # Load the zarr file
    z = zarr.open(zarr_path, mode='r')
    print(z.tree())

    group_names = [group_name for group_name, store in z.groups()]

    num_files = len(hr_split_indices) + 1  # Create n+1 files per n splits

    # Build an explicit {group: base_split_indices} mapping (REG derived from shapes)
    split_indices_map = build_split_indices_map(
        z, group_names, hr_split_indices, lr_split_indices, split_axis
    )

    # Validate: a group is either not split (empty) or split into exactly num_files parts
    for group_name, base_indices in split_indices_map.items():
        if len(base_indices) not in (0, num_files - 1):
            raise ValueError(
                f"Group '{group_name}' has {len(base_indices)} split indices; "
                f"expected 0 or {num_files - 1} to match hr_split_indices."
            )

    for i in range(num_files):
        split_path = out_ome_path.replace(".zarr", f"_{i}.zarr")
        print(f"\n Writing split file {i + 1}/{num_files}: {split_path}")

        for g, group_name in enumerate(group_names):
            base_indices = split_indices_map[group_name]
            depth = len(z[group_name])

            # First group opens the file for (over)writing, the rest append
            mode = "w" if g == 0 else "a"
            store, _ = create_ome_group(
                split_path, group_name=group_name, pyramid_depth=depth, mode=mode
            )

            # Copy / split each pyramid level lazily via dask (no full materialization)
            for level in range(depth):
                arr = da.from_zarr(z[f"{group_name}/{level}"])
                parts = split_level(arr, base_indices, level, split_axis, num_files)
                write_ome_level(
                    parts[i], store, group_name,
                    level=level, chunk_size=tuple(args.chunk_size), cname='lz4',
                )

        # Print tree to verify splits
        z_tmp = zarr.open(split_path, mode='r')
        print("\n Writing split file finished.")
        print(z_tmp.tree())

    print("Done")