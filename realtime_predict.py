import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np

class IOStreamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Input → Output Volume Meter")

        # Query all devices
        all_devices = sd.query_devices()

        # Separate input vs. output devices
        self.input_devices = [
            (idx, dev["name"])
            for idx, dev in enumerate(all_devices)
            if dev["max_input_channels"] > 0
        ]
        self.output_devices = [
            (idx, dev["name"])
            for idx, dev in enumerate(all_devices)
            if dev["max_output_channels"] > 0
        ]

        # --- GUI Elements ---

        # 1) Input device selection
        self.input_label = tk.Label(root, text="Select Input Device:")
        self.input_label.pack(pady=(10, 0))

        self.input_var = tk.StringVar()
        self.input_combobox = ttk.Combobox(root, textvariable=self.input_var, state="readonly")
        self.input_combobox["values"] = [f"[{d[0]}] {d[1]}" for d in self.input_devices]
        if self.input_devices:
            self.input_combobox.current(0)  # default to first
        self.input_combobox.pack(pady=5, padx=10)

        # 2) Output device selection
        self.output_label = tk.Label(root, text="Select Output Device:")
        self.output_label.pack(pady=(10, 0))

        self.output_var = tk.StringVar()
        self.output_combobox = ttk.Combobox(root, textvariable=self.output_var, state="readonly")
        self.output_combobox["values"] = [f"[{d[0]}] {d[1]}" for d in self.output_devices]
        if self.output_devices:
            self.output_combobox.current(0)  # default to first
        self.output_combobox.pack(pady=5, padx=10)

        # 3) Start / Stop Buttons
        self.start_button = tk.Button(root, text="Start", command=self.start_stream)
        self.start_button.pack(side=tk.LEFT, padx=(50, 5), pady=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_stream, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=10)

        # 4) Volume label
        self.volume_label = tk.Label(root, text="Volume: 0.000")
        self.volume_label.pack(pady=(10, 0))

        # 5) Volume progress bar (0 to 1.0 range)
        self.volume_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.volume_bar["value"] = 0
        self.volume_bar["maximum"] = 1.0
        self.volume_bar.pack(pady=5)

        # Internal variables
        self.stream = None
        self.latest_volume = 0.0

        # Schedule periodic GUI update
        self.update_gui()

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print("[SoundDevice Callback Warning]", status)

        # Measure peak amplitude of input signal
        # indata shape: (frames, channels). We'll take channel 0 for simplicity
        mono_in = indata[:, 0]
        peak = np.abs(mono_in).max()
        self.latest_volume = peak

        # Pass input directly to output (mono)
        outdata[:, 0] = mono_in

    def start_stream(self):
        """
        Start a sounddevice.Stream using the selected input and output.
        """
        in_sel = self.input_combobox.current()
        out_sel = self.output_combobox.current()
        if in_sel < 0 or out_sel < 0:
            return  # invalid selection

        input_device_index = self.input_devices[in_sel][0]
        output_device_index = self.output_devices[out_sel][0]

        # Lock the comboboxes, disable Start, enable Stop
        self.input_combobox.config(state=tk.DISABLED)
        self.output_combobox.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Create and start the stream
        try:
            self.stream = sd.Stream(
                device=(input_device_index, output_device_index),
                samplerate=44100,
                blocksize=1024,
                dtype='float32',
                channels=1,  # single channel in & out for simplicity
                callback=self.audio_callback
            )
            self.stream.start()
        except Exception as e:
            print("Error opening I/O stream:", e)
            # Re-enable Start in case of error
            self.input_combobox.config(state="readonly")
            self.output_combobox.config(state="readonly")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def stop_stream(self):
        """
        Stop the audio stream.
        """
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Re-enable device selection and Start
        self.input_combobox.config(state="readonly")
        self.output_combobox.config(state="readonly")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_gui(self):
        """
        Periodically update the volume label and bar on the main thread.
        """
        self.volume_label.config(text=f"Volume: {self.latest_volume:.3f}")
        self.volume_bar["value"] = self.latest_volume

        # Schedule next update in 50 ms
        self.root.after(50, self.update_gui)

def main():
    root = tk.Tk()
    app = IOStreamGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
