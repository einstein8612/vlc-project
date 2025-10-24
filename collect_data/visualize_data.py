import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DataViewerGUI:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.data_files = sorted([name for name in os.listdir(input_dir) if name.endswith('.npy')])
        self.index = 0
        self.total = len(self.data_files)

        # Setup Tkinter window
        self.root = tk.Tk()
        self.root.title("Data Viewer GUI")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.lines = [self.ax.plot([], [], label=f"PD {i+1}")[0] for i in range(4)]
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("PD Value")
        self.ax.set_title("Photodiode Values Over Time")
        self.ax.legend()
        self.ax.grid(True)

        # Incoming data figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Next button
        self.next_button = tk.Button(self.root, text="NEXT", bg="green", fg="white",
                                       font=("Arial", 16), command=self.next_clicked)
        self.next_button.pack(side=tk.TOP, pady=10)

        # Previous button
        self.previous_button = tk.Button(self.root, text="PREVIOUS", bg="red", fg="white",
                                       font=("Arial", 16), command=self.prev_clicked)
        self.previous_button.pack(side=tk.TOP, pady=10)

        # Dataset counter label
        self.count_label = tk.Label(self.root, text=f"Index: {self.index+1}/{self.total}",
                                    font=("Arial", 14))
        self.count_label.pack(side=tk.TOP, pady=5)

        # Start acquisition loop
        self.root.after(100, self.draw)
        self.root.mainloop()
    
    def draw(self):
        data_path = os.path.join(self.input_dir, self.data_files[self.index])
        data = np.load(data_path)

        time_axis = np.arange(data.shape[0]) * 0.01  # Assuming 100 Hz sampling rate

        for i in range(4):
            self.lines[i].set_data(time_axis, data[:, i])

        self.ax.relim()
        self.ax.autoscale_view()

        self.count_label.config(text=f"Index: {self.index+1}/{self.total}")

        self.canvas.draw()

    def next_clicked(self):
        if self.index < self.total - 1:
            self.index += 1
            self.draw()

    def prev_clicked(self):
        if self.index > 0:
            self.index -= 1
            self.draw()

    def on_close(self):
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "collected_data"

    DataViewerGUI(input_dir)
