import numpy as np
import os
from tqdm import tqdm
from scipy.interpolate import CubicSpline

def time_shift(x, shift_max=50):
    """Randomly shift the sequence along the time axis"""
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(x, shift, axis=0)

def time_warp(x, sigma=0.2):
    """Apply smooth random time warping to the sequence"""
    orig_steps = np.arange(x.shape[0])
    random_curve = np.cumsum(np.random.normal(loc=1.0, scale=sigma, size=x.shape[0]))
    warped_steps = (random_curve - random_curve.min()) / (random_curve.max() - random_curve.min()) * (x.shape[0]-1)
    
    warped = np.zeros_like(x)
    for i in range(x.shape[1]):
        cs = CubicSpline(warped_steps, x[:, i])
        warped[:, i] = cs(orig_steps)
    return warped

def augment_data(x):
    x_aug = time_shift(x)
    x_aug = time_warp(x_aug)
    return x_aug

def augment_dataset(root_dir, output_dir, fraction_real=0.2):
    os.makedirs(output_dir, exist_ok=True)
    id_folders = sorted(os.listdir(root_dir))

    for id_name in tqdm(id_folders, desc="Augmenting data"):
        id_path = os.path.join(root_dir, id_name)
        out_id_path = os.path.join(output_dir, id_name)
        os.makedirs(out_id_path, exist_ok=True)
        
        file_list = [f for f in os.listdir(id_path) if f.endswith('.npy')]
        np.random.shuffle(file_list)
        num_real = int(len(file_list) * fraction_real)

        for f_name in file_list[:num_real]:
            data = np.load(os.path.join(id_path, f_name))
            np.save(os.path.join(out_id_path, f_name), data)

        for f_name in file_list[num_real:]:
            data = np.load(os.path.join(id_path, f_name))
            augmented_data = augment_data(data)
            aug_name = f_name.replace('.npy', '_aug.npy')
            np.save(os.path.join(out_id_path, aug_name), augmented_data)

if __name__ == "__main__":
    augment_dataset('preprocessed_data', 'augmented_data_80', fraction_real=0.8)