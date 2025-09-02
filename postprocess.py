import os
import glob
import zarr
import numpy as np
import matplotlib.pyplot as plt

import dask.array as da
from utils.utils_image import match_histogram_3d_continuous, plot_histogram, compare_histograms
from utils.utils_plot import viz_slices, viz_multiple_images

if __name__ == '__main__':

    #data_path = "../3D_datasets/datasets/"
    data_path = "../Vedrana_master_project/3D_datasets/datasets/"

    train_paths = {"HCP_1200": glob.glob(os.path.join(data_path, "HCP_1200/ome/train/*.zarr")),
                   "IXI": glob.glob(os.path.join(data_path, "IXI/ome/train/*.zarr")),
                   "LITS": glob.glob(os.path.join(data_path, "LITS/ome/train/*.zarr")),
                   "CTSpine1K": glob.glob(os.path.join(data_path, "CTSpine1K/ome/train/*.zarr")),
                   "LIDC-IDRI": glob.glob(os.path.join(data_path, "LIDC_IDRI/ome/train/*.zarr")),
                   "VoDaSuRe": glob.glob(os.path.join(data_path, "VoDaSuRe/ome/train/*.zarr"))}

    test_paths = {"HCP_1200": glob.glob(os.path.join(data_path, "HCP_1200/ome/test/*.zarr")),
                  "IXI": glob.glob(os.path.join(data_path, "IXI/ome/test/*.zarr")),
                  "LITS": glob.glob(os.path.join(data_path, "LITS/ome/test/*.zarr")),
                  "CTSpine1K": glob.glob(os.path.join(data_path, "CTSpine1K/ome/test/*.zarr")),
                  "LIDC-IDRI": glob.glob(os.path.join(data_path, "LIDC_IDRI/ome/test/*.zarr")),
                  "VoDaSuRe": glob.glob(os.path.join(data_path, "VoDaSuRe/ome/test/*.zarr"))}

    print("test")

    z = zarr.open(train_paths['HCP_1200'][0], mode='a+')

    for group_name in z:
        print(f"Group name: {group_name}")
        for level in z[group_name]:
            if level == '0':
                if group_name == 'REG':
                    ref = z['HR'][level]  # Select HR as reference
                else:
                    ref = z[group_name][level]  # Select lowest level image in group as reference
                    continue

            # Read source image
            src = z[group_name][level]

            # compute matched histogram
            matched = match_histogram_3d_continuous(source=src, reference=ref)
            compare_histograms(ref, matched)

            # Write back to zarr
            src = matched







    # compare_histograms(z['HR']['0'], z['HR']['2'])
    # compare_histograms(z['HR']['0'], matched)
    # viz_slices(z['HR']['2'], [20, 40, 60], savefig=False)
    # viz_slices(matched, [20, 40, 60], savefig=False)
    # viz_slices(z['HR']['0'], [20*4, 40*4, 60*4], savefig=False)


