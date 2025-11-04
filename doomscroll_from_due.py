import serial
import sys
import struct
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
import time
import json

from dataset import InMemoryCNNLSTMDataset
from model import CNNLSTM  # your CNNLSTM class

from doomscroll_interface import (
    start_doomscroll,
    scroll_up,
    scroll_down,
    toggle_play,
    like,
    close_driver,
)

class DoomscrollGUI:
    def __init__(self, port, config, model_path, device="cpu", labels=None):
        self.port = port
        self.device = device
        self.labels = labels or [f"Class {i}" for i in range(9)]
        self.pred_label = "N/A"
        self.config = config

        self.factor = 4 # Downsampling factor

        # Load model
        self.model = CNNLSTM(num_classes=len(self.labels), device=device).to(device)
        self.model.load(model_path)
        self.model.eval()
        
        # Load doomscroll module
        start_doomscroll(mode="selenium", browser="firefox", headless=False, wait=6)

        # Connect to serial port
        # try:
        #     self.ser = serial.Serial(port, 115200, timeout=None)
        #     print(f"Connected to {port}")
        # except serial.SerialException as e:
        #     print(f"Error opening serial port {port}: {e}")
        #     sys.exit(1)

        # # Read Arduino settings
        # self.init_config()

        print("Starting real-time data acquisition...")
        self.run()

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
        
        time.sleep(4)
        self.run_actions("right_to_left")
        time.sleep(4)
        self.run_actions("right_to_left")
        time.sleep(4)
        self.run_actions("double_tap")
        time.sleep(4)
        self.run_actions("right_to_left")
        time.sleep(4)
        self.run_actions("right_to_left")
        time.sleep(4)
        self.run_actions("right_to_left")
        time.sleep(4)
        
        # Read sample
        total_samples = self.sample_length * self.fs
        pd_values = np.zeros((total_samples, 4), dtype=np.float32)

        for i in range(total_samples):
            sample_data = self.ser.read(8)
            for ch in range(4):
                pd_values[i, ch] = struct.unpack(
                    "<H", sample_data[ch * 2 : ch * 2 + 2]
                )[0]

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

        # Classify current data
        label, pred_idx, pred_conf = self.classify_data(processed_pd_values)
        print(f"Predicted gesture {label} ({pred_idx}) with confidence {pred_conf:.2f}")
        
        if pred_conf > self.config["confidence_threshold"]:
            self.run_actions(label)
        
        self.run()

    def run_actions(self, label):
        for action, action_label in self.config["actions"].items():
            if label != action_label:
                continue
            
            if action == "scroll_up":
                scroll_up()
            elif action == "scroll_down":
                scroll_down()
            elif action == "toggle_play":
                toggle_play()
            elif action == "like":
                like()
            return

    def classify_data(self, pd_values):
        with torch.no_grad():
            x = torch.tensor(
                pd_values, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, T, 4)
            logits = self.model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_conf = torch.softmax(logits, dim=1)[pred_idx].item()
            return self.labels[pred_idx], pred_idx, pred_conf

    def on_close(self):
        self.ser.close()
        close_driver()
        sys.exit(0)

def read_doomscroll_config():
    with open("doomscroll_actions.json", "rb") as file:
        return json.load(file)
    return None

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
    
    config = read_doomscroll_config()

    DoomscrollGUI(port, config, model_path, device=device, labels=labels)
