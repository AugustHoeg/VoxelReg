import numpy as np
import qim3d as qim

def load_txm(input_path, dtype=np.float32):
    print(f"Reading input file: {input_path}")
    image = qim.io.load(input_path, virtual_stack=False, progress_bar=True, dtype=dtype)
    print(f"txm shape: {image.shape}")
    return image