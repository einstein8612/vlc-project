import serial
import sys
import struct
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch

from dataset import InMemoryCNNLSTMDataset
from model import CNNLSTM  # your CNNLSTM class


class ClassifyGUI:
    def __init__(self, port, model_path, device="cpu", labels=None):
        self.port = port
        self.device = device
        self.labels = labels or [f"Class {i}" for i in range(9)]
        self.current_data = None
        self.pred_label = "N/A"

        self.factor = 4 # Downsampling factor

        # Load model
        self.model = CNNLSTM(num_classes=len(self.labels), device=device).to(device)
        self.model.load(model_path)
        self.model.eval()

        # Connect to serial port
        try:
            self.ser = serial.Serial(port, 115200, timeout=None)
            print(f"Connected to {port}")
        except serial.SerialException as e:
            print(f"Error opening serial port {port}: {e}")
            sys.exit(1)

        # Read Arduino settings
        self.init_config()

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Real-time Classification GUI")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Plot setup
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.lines = [self.ax.plot([], [], label=f"PD {i + 1}")[0] for i in range(4)]
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("PD Value")
        self.ax.set_title("Photodiode Values Over Time")
        self.ax.set_xlim(0, self.sample_length)
        self.ax.set_ylim(0, 1)
        self.ax.legend()
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Label to show prediction
        self.pred_label_widget = tk.Label(
            self.root,
            text=f"Prediction: {self.pred_label}",
            font=("Arial", 16),
            bg="yellow",
        )
        self.pred_label_widget.pack(side=tk.TOP, pady=10)

        print("Starting real-time data acquisition...")
        self.root.after(100, self.run)
        self.root.mainloop()

    def init_config(self):
        data = self.ser.read(8)
        self.duration = struct.unpack("B", data[0:1])[0]
        self.fs = struct.unpack("<H", data[1:3])[0]
        self.sample_length = struct.unpack("B", data[3:4])[0]
        self.delay_us = struct.unpack("<I", data[4:8])[0]

        print(
            f"Duration: {self.duration}, Fs: {self.fs}, Sample length: {self.sample_length}, Delay: {self.delay_us}"
        )

    def run(self):
        if self.ser.in_waiting == 0:
            self.root.after(50, self.run)
            return

        # Read sample
        total_samples = self.sample_length * self.fs
        pd_values = np.zeros((total_samples, 4), dtype=np.float32)

        for i in range(total_samples):
            sample_data = self.ser.read(8)
            for ch in range(4):
                pd_values[i, ch] = struct.unpack(
                    "<H", sample_data[ch * 2 : ch * 2 + 2]
                )[0]

        self.current_data = pd_values

        processed_pd_values = np.zeros((total_samples // self.factor, 4), dtype=np.float32)

        # Filter PD values
        for ch in range(4):
            y = pd_values[:, ch]
            y = y[::self.factor]
            processed_pd_values[:, ch] = savgol_filter(
                y, window_length=101, polyorder=3, mode="interp"
            )
        
        # Min-max normalization
        processed_pd_values = (processed_pd_values - np.min(processed_pd_values, axis=0)) / \
            (np.max(processed_pd_values, axis=0) - np.min(processed_pd_values, axis=0) + 1e-6)

        # Update plot
        time_axis = np.arange(total_samples / self.factor) / self.fs * self.factor
        for ch in range(4):
            y = processed_pd_values[:, ch]
            self.lines[ch].set_data(time_axis, y)

        self.ax.set_xlim(0, self.sample_length)
        self.ax.set_ylim(0, 1)
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Classify current data
        self.classify_data(processed_pd_values)

        self.root.after(50, self.run)

    def classify_data(self, pd_values):
        with torch.no_grad():
            x = torch.tensor(
                pd_values, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, T, 4)
            logits = self.model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
            print(pred_idx)
            self.pred_label = self.labels[pred_idx]
            self.pred_label_widget.config(text=f"Prediction: {self.pred_label}")

    def on_close(self):
        self.ser.close()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    port = (
        sys.argv[1] if len(sys.argv) > 1 else input("Enter serial port (e.g., COM3): ")
    )
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = InMemoryCNNLSTMDataset(
        root_dir="dataset/preprocessed_data",
        ignored_gesture_ids=[],
        device=device
    )
    labels = dataset.get_labels()
    del dataset
    labels = {v: k for k, v in labels.items()} # ID to name mapping


    ClassifyGUI(port, model_path, device=device, labels=labels)
