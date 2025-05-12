import os
import itk
import nibabel as nib
import numpy as np
import argparse
from utils.utils_tiff import load_tiff

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split volume and save")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--scan_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--out_path", type=str, required=False, help="Path to the output file.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")
    parser.add_argument("--chunk_size", type=int, nargs=3, default=(648, 648, 648), help="Size of each chunk (D, H, W).")

    args = parser.parse_args()
    return args

def chunk_indices(image_shape, chunk_size):
    """Yield start indices for chunking along each dimension."""
    for z in range(0, image_shape[0], chunk_size[0]):
        for y in range(0, image_shape[1], chunk_size[1]):
            for x in range(0, image_shape[2], chunk_size[2]):
                yield (z, y, x)

def get_chunk(image, start_idx, chunk_size):
    """Slice a chunk from the image at the given start index."""
    z, y, x = start_idx
    z_end = min(z + chunk_size[0], image.shape[0])
    y_end = min(y + chunk_size[1], image.shape[1])
    x_end = min(x + chunk_size[2], image.shape[2])
    return image[z:z_end, y:y_end, x:x_end]


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
    if args.scan_path is not None:
        scan_path = os.path.join(sample_path, args.scan_path)
        print("Scan path: ", scan_path)
    if args.out_path is not None:
        out_path = os.path.join(sample_path, args.out_path)  # os.path.join(sample_path, args.out_name)
        print("Output path: ", out_path)
    if args.out_name is not None:
        out_name = args.out_name  # os.path.join(sample_path, args.out_name)
        print("Output name: ", out_name)

    print(f"Loading {scan_path}")

    filename, file_extension = os.path.basename(scan_path).split('.', 1)

    if file_extension == "nii" or file_extension == "nii.gz":
        image = itk.imread(scan_path).astype(np.float32)

    elif file_extension == "tiff" or file_extension == "tif":
        image = load_tiff(scan_path).astype(np.float32)

    elif file_extension == "npy":
        image = np.load(scan_path).astype(np.float32)

    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    # image = np.load(scan_path).astype(np.float32)
    # If you must open large images, consider using memory-mapped arrays:
    # image = np.load("image.npy", mmap_mode='r')  # won't load entire array into RAM

    chunk_size = args.chunk_size

    print("Image shape: ", image.shape)
    print("Chunk size: ", chunk_size)

    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    for idx, start_idx in enumerate(chunk_indices(image.shape, chunk_size)):
        chunk = get_chunk(image, start_idx, chunk_size)
        print(f"Processing chunk {idx} at {start_idx} with shape {chunk.shape}")

        # Save / process the chunk
        np.save(os.path.join(out_path, f"{out_name}_chunk_{idx}.npy"), chunk)


