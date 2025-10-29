import numpy as np
import os

from tqdm import tqdm

def downsample(root_dir, output_dir, factor=4):
    os.makedirs(output_dir, exist_ok=True)

    id_folders = sorted(os.listdir(root_dir))
    for id_name in tqdm(id_folders, desc="Downsampling data"):
        id_path = os.path.join(root_dir, id_name)
        os.makedirs(os.path.join(output_dir, id_name), exist_ok=True)
        for f_name in os.listdir(id_path):
            if f_name.endswith('.npy'):
                f_path = os.path.join(id_path, f_name)
                o_path = os.path.join(output_dir, id_name, f_name)
                data = np.load(f_path)  # shape: (T, 4)
                # Downsample by taking every f-th sample
                downsampled_data = data[::factor]
                np.save(o_path, downsampled_data)

downsample('data', 'downsampled_data', factor=4)