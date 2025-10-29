import numpy as np
import os

from scipy.signal import savgol_filter
from tqdm import tqdm

def preprocess_signal(data):
    filtered_data = np.zeros_like(data, dtype=np.float32)
    for ch in range(data.shape[1]):
        filtered_ch = savgol_filter(data[:, ch], window_length=101, polyorder=2, mode='interp')
        filtered_ch = (filtered_ch - filtered_ch.min()) / (filtered_ch.max() - filtered_ch.min() + 1e-9)
        filtered_data[:, ch] = filtered_ch
    return filtered_data

def preprocess_folder(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    id_folders = sorted(os.listdir(root_dir))
    for id_name in tqdm(id_folders, desc="Preprocessing data"):
        id_path = os.path.join(root_dir, id_name)
        os.makedirs(os.path.join(output_dir, id_name), exist_ok=True)
        for f_name in os.listdir(id_path):
            if f_name.endswith('.npy'):
                f_path = os.path.join(id_path, f_name)
                o_path = os.path.join(output_dir, id_name, f_name)
                data = np.load(f_path)  # shape: (T, 4)
                filtered_data = preprocess_signal(data)
                np.save(o_path, filtered_data)

preprocess_folder('downsampled_data', 'preprocessed_data')