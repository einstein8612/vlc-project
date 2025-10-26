import os
import serial
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DataCollectGUI:
    def __init__(self, port, output_dir):
        self.port = port
        self.output_dir = output_dir
        self.current_data = None
        self.rejected_flag = True
        self.new_data_count = 0

        # Connect to serial port
        try:
            self.ser = serial.Serial(port, 115200, timeout=None)
            print(f"Connected to {port}")
        except serial.SerialException as e:
            print(f"Error opening serial port {port}: {e}")
            sys.exit(1)

        # Read Arduino settings 
        self.init_config()

        # Setup Tkinter window
        self.root = tk.Tk()
        self.root.title("Data Collection GUI")
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

        # Accept button
        self.accept_button = tk.Button(self.root, text="ACCEPT", bg="green", fg="white",
                                       font=("Arial", 16), command=self.accept_clicked)
        self.accept_button.pack(side=tk.TOP, pady=10)

        # Reject button
        self.reject_button = tk.Button(self.root, text="REJECT", bg="red", fg="white",
                                       font=("Arial", 16), command=self.reject_clicked)
        self.reject_button.pack(side=tk.TOP, pady=10)

        # Dataset counter label
        self.count_label = tk.Label(self.root, text=f"Data Collected: {self.new_data_count}",
                                    font=("Arial", 14))
        self.count_label.pack(side=tk.TOP, pady=5)

        print("Press button to start samples...")
        # Start acquisition loop
        self.root.after(100, self.run)
        self.root.mainloop()

    def init_config(self):
        data = self.ser.read(8)  # u8 + u16 + u8 + u32
        self.duration = struct.unpack('B', data[0:1])[0]
        self.fs = struct.unpack('<H', data[1:3])[0]
        self.sample_length = struct.unpack('B', data[3:4])[0]
        self.delay_us = struct.unpack('<I', data[4:8])[0]

        print(f"Duration: {self.duration}")
        print(f"Sampling Frequency: {self.fs}")
        print(f"Sample Length: {self.sample_length}")
        print(f"Delay (us): {self.delay_us}")

    def accept_clicked(self):
        if self.rejected_flag or self.current_data is None:
            return
        self.new_data_count += 1
        filename = os.path.join(self.output_dir, f"data_{self.new_data_count:05d}.npy")
        np.save(filename, self.current_data)
        self.count_label.config(text=f"Data Collected: {self.new_data_count}")

    def reject_clicked(self):
        for ch in range(4):
            self.lines[ch].set_data([], [])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.rejected_flag = True
        self.current_data = None

    def run(self):
        # Check if data is available
        if (self.ser.in_waiting == 0):
            self.root.after(100, self.run)
            return
        
        # Reject or accept data when new data comes in
        if not self.rejected_flag:
            self.accept_clicked()

        self.rejected_flag = False

        # Prepare to read data
        total_samples = self.sample_length * self.fs
        pd_values = np.zeros((total_samples, 4), dtype=np.uint16)

        # Read sample data
        for i in range(total_samples):
            sample_data = self.ser.read(8)
            for ch in range(4):
                pd_values[i, ch] = struct.unpack('<H', sample_data[ch*2:ch*2+2])[0]
        self.current_data = pd_values 

        # Update plot
        time_axis = np.arange(total_samples) / self.fs
        for ch in range(4):
            self.lines[ch].set_data(time_axis, pd_values[:, ch])
        self.ax.set_xlim(0, self.duration)
        self.ax.set_ylim(0, 1024)
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Update counter display
        self.count_label.config(text=f"Data Collected: {self.new_data_count}")

        self.root.after(100, self.run)

    def on_close(self):
        self.ser.close()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    port = (
        sys.argv[1]
        if len(sys.argv) > 1
        else input("Enter the serial port (e.g., COM3 or /dev/ttyUSB0): ")
    )

    output_dir = sys.argv[2] if len(sys.argv) > 2 else "collected_data"
    if os.path.exists(output_dir):
        print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING")
        print(f"Output directory '{output_dir}' already exists. This will overwrite existing data.")
        print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING")
    os.makedirs(output_dir, exist_ok=True)

    DataCollectGUI(port, output_dir)
