import torch

from timegan_lib.options import Options
from timegan_lib.lib.timegan import TimeGAN
from dataset import InMemoryCNNLSTMDataset

LOAD = "timegan_gesture_model_left_to_right.pth"

if not LOAD:
    dataset = InMemoryCNNLSTMDataset(root_dir="preprocessed_data")
    labels_tensor = torch.stack(dataset.labels)
    selected_samples = [s for s, l in zip(dataset.samples, labels_tensor) if l == 4]

    timegan = TimeGAN(opt=Options().parse(), ori_data=selected_samples)
    timegan.train()

    timegan.save("timegan_gesture_model_left_to_right.pth")

timegan = TimeGAN(opt=Options().parse())
timegan.load(LOAD)

synthetic_data = timegan.generate_synthetic(num_samples=10, seq_len=500)
print(synthetic_data.shape)
torch.save(synthetic_data, "synthetic_gesture_data.pt") # Manually postprocess to .npy files
